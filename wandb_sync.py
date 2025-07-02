#!/usr/bin/env python3
"""
高级 WandB offline 同步脚本，支持：
- 自动监控新的 offline runs
- 同步状态持久化
- 失败重试机制
- 详细的日志和报告
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import signal
import sys
from typing import Dict, List, Tuple
import hashlib

# 设置日志
def setup_logging(log_file=None):
    """设置日志配置"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


class WandBSyncManager:
    """WandB 同步管理器"""
    
    def __init__(self, base_path: str, state_file: str = None):
        self.base_path = Path(base_path)
        self.state_file = state_file or self.base_path / ".wandb_sync_state.json"
        self.state = self.load_state()
        self.logger = logging.getLogger(__name__)
        self.running = True
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """处理中断信号"""
        self.logger.info("收到中断信号，正在优雅退出...")
        self.running = False
        self.save_state()
        sys.exit(0)
    
    def load_state(self) -> Dict:
        """加载同步状态"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "synced": {},
            "failed": {},
            "last_scan": None
        }
    
    def save_state(self):
        """保存同步状态"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_run_hash(self, run_path: Path) -> str:
        """获取运行的唯一标识"""
        # 使用路径和修改时间生成哈希
        stat = run_path.stat()
        content = f"{run_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def find_new_runs(self) -> List[Path]:
        """查找新的或未同步的 offline runs"""
        all_runs = []
        new_runs = []
        
        # 搜索所有 offline-run 目录
        for run_path in self.base_path.rglob("offline-run-*"):
            if not run_path.is_dir():
                continue
                
            all_runs.append(run_path)
            run_id = str(run_path)
            run_hash = self.get_run_hash(run_path)
            
            # 检查是否是新的或已更改的运行
            if run_id not in self.state["synced"] or \
               self.state["synced"][run_id].get("hash") != run_hash:
                new_runs.append(run_path)
        
        self.logger.info(f"总共找到 {len(all_runs)} 个运行，其中 {len(new_runs)} 个需要同步")
        return new_runs
    
    def sync_single_run(self, run_path: Path, retry_count: int = 3) -> Tuple[bool, str]:
        """同步单个运行，支持重试"""
        run_id = str(run_path)
        
        for attempt in range(retry_count):
            try:
                self.logger.info(f"同步运行 (尝试 {attempt + 1}/{retry_count}): {run_path.name}")
                
                # 直接使用完整路径执行 wandb sync
                result = subprocess.run(
                    ["wandb", "sync", str(run_path)],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10分钟超时
                )
                
                if result.returncode == 0:
                    # 同步成功
                    self.state["synced"][run_id] = {
                        "hash": self.get_run_hash(run_path),
                        "sync_time": datetime.now().isoformat(),
                        "attempts": attempt + 1
                    }
                    # 从失败列表中移除（如果存在）
                    self.state["failed"].pop(run_id, None)
                    
                    # 解析输出以获取同步的URL
                    if "Syncing:" in result.stdout:
                        url_line = [line for line in result.stdout.split('\n') if "Syncing:" in line]
                        if url_line:
                            self.logger.info(f"✓ 成功同步: {run_path.name}")
                            self.logger.info(f"  URL: {url_line[0].strip()}")
                    else:
                        self.logger.info(f"✓ 成功同步: {run_path.name}")
                    
                    return True, result.stdout
                else:
                    self.logger.warning(f"同步失败 (尝试 {attempt + 1}): {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"同步超时 (尝试 {attempt + 1}): {run_path.name}")
            except Exception as e:
                self.logger.error(f"同步异常 (尝试 {attempt + 1}): {e}")
            
            # 重试前等待
            if attempt < retry_count - 1:
                time.sleep(5 * (attempt + 1))
        
        # 所有尝试都失败
        self.state["failed"][run_id] = {
            "last_attempt": datetime.now().isoformat(),
            "attempts": retry_count,
            "error": "Max retries exceeded"
        }
        return False, "All retry attempts failed"
    
    def sync_runs(self, runs: List[Path], max_workers: int = 4):
        """并行同步多个运行"""
        if not runs:
            self.logger.info("没有需要同步的运行")
            return
        
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {
                executor.submit(self.sync_single_run, run): run 
                for run in runs
            }
            
            for future in as_completed(future_to_run):
                if not self.running:
                    break
                    
                success, message = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                
                # 定期保存状态
                if (successful + failed) % 5 == 0:
                    self.save_state()
        
        # 保存最终状态
        self.save_state()
        
        # 打印统计
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"同步统计:")
        self.logger.info(f"  成功: {successful}")
        self.logger.info(f"  失败: {failed}")
        self.logger.info(f"  总计: {len(runs)}")
        self.logger.info(f"{'='*50}")
    
    def watch_and_sync(self, interval: int = 300, max_workers: int = 4):
        """监控模式：定期检查并同步新的运行"""
        self.logger.info(f"开始监控模式，每 {interval} 秒检查一次")
        
        while self.running:
            try:
                # 查找新运行
                new_runs = self.find_new_runs()
                
                # 同步新运行
                if new_runs:
                    self.sync_runs(new_runs, max_workers)
                
                # 更新最后扫描时间
                self.state["last_scan"] = datetime.now().isoformat()
                self.save_state()
                
                # 等待下一次扫描
                self.logger.info(f"等待 {interval} 秒后进行下一次扫描...")
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"监控循环出错: {e}")
                time.sleep(30)  # 出错后等待30秒
    
    def get_status_report(self) -> str:
        """生成状态报告"""
        report = []
        report.append(f"\nWandB 同步状态报告")
        report.append(f"{'='*50}")
        report.append(f"基础路径: {self.base_path}")
        report.append(f"最后扫描: {self.state.get('last_scan', 'Never')}")
        report.append(f"已同步运行: {len(self.state['synced'])}")
        report.append(f"失败运行: {len(self.state['failed'])}")
        
        if self.state['failed']:
            report.append(f"\n失败的运行:")
            for run_id, info in self.state['failed'].items():
                report.append(f"  - {run_id}")
                report.append(f"    最后尝试: {info['last_attempt']}")
                report.append(f"    尝试次数: {info['attempts']}")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="高级 WandB offline 同步工具")
    parser.add_argument(
        "--path",
        default="/mnt/petrelfs/lihaozhan/Rob/SimpleVLA-RL-robotwin-prop/wandb",
        help="WandB 目录路径"
    )
    parser.add_argument(
        "--mode",
        choices=["once", "watch"],
        default="once",
        help="运行模式: once (单次同步) 或 watch (持续监控)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="监控模式下的检查间隔（秒）"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行工作线程数"
    )
    parser.add_argument(
        "--log-file",
        help="日志文件路径"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="显示同步状态报告"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置同步状态"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_file)
    
    # 创建同步管理器
    manager = WandBSyncManager(args.path)
    
    # 处理不同的命令
    if args.status:
        print(manager.get_status_report())
        return
    
    if args.reset:
        manager.state = {"synced": {}, "failed": {}, "last_scan": None}
        manager.save_state()
        logger.info("同步状态已重置")
        return
    
    # 检查 wandb 登录状态
    try:
        result = subprocess.run(["wandb", "login", "--verify"], capture_output=True)
        if result.returncode != 0:
            logger.error("WandB 未登录，请先运行 'wandb login'")
            return
    except:
        logger.error("无法运行 wandb 命令，请确保已安装 wandb")
        return
    
    # 执行同步
    if args.mode == "once":
        new_runs = manager.find_new_runs()
        manager.sync_runs(new_runs, args.workers)
    else:
        manager.watch_and_sync(args.interval, args.workers)


if __name__ == "__main__":
    main()