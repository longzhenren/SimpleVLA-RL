
#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 拼接目标路径
BASE_PATH="${SCRIPT_DIR}/verl/utils/envs/robotwin2/envs/robot"

# 检查目标目录是否存在
if [ ! -d "$BASE_PATH" ]; then
    echo "错误: 目标目录不存在: $BASE_PATH"
    exit 1
fi

# 切换到目标目录
cd "$BASE_PATH" || exit 1

echo " rename planner.py -> planner_mplib.py and planner_curobo.py -> planner.py"

# 1. 重命名 planner.py -> planner_mplib.py
if [ -f "planner.py" ]; then
    mv "planner.py" "planner_mplib.py"
    echo "✓ planner.py -> planner_mplib.py"
else
    echo "✗ planner.py 不存在"
fi

# 2. 重命名 robot.py -> robot_mplib.py
if [ -f "robot.py" ]; then
    mv "robot.py" "robot_mplib.py"
    echo "✓ robot.py -> robot_mplib.py"
else
    echo "✗ robot.py 不存在"
fi

# 3. 重命名 robot_curobo.py -> robot.py
if [ -f "robot_curobo.py" ]; then
    mv "robot_curobo.py" "robot.py"
    echo "✓ robot_curobo.py -> robot.py"
else
    echo "✗ robot_curobo.py 不存在"
fi

# 4. 重命名 planner_curobo.py -> planner.py
if [ -f "planner_curobo.py" ]; then
    mv "planner_curobo.py" "planner.py"
    echo "✓ planner_curobo.py -> planner.py"
else
    echo "✗ planner_curobo.py 不存在"
fi

echo "Now use curobo planner for pre-collect seed."

#start collect seed
DATASET_NAME="robotwin2.0 task name"
cd "${SCRIPT_DIR}/verl/workers/rollout"
python pre_collect_twin2_seed.py --tasks $DATASET_NAME --seed-start  100000  --seed-end  200000  --target-count 1000  --num-gpus 8  --data-split train
#python pre_collect_twin2_seed.py --tasks $DATASET_NAME --seed-start  100000000  --seed-end  100100000  --target-count 160  --num-gpus 8  --data-split eval
# collect seed end 

#!/bin/bash

cd "$BASE_PATH" || exit 1

echo " rename planner.py -> planner_curobo.py and planner_mplib.py -> planner.py "

# 反向操作需要按相反顺序执行，避免文件名冲突

# 1. 重命名 planner.py -> planner_curobo.py
if [ -f "planner.py" ]; then
    mv "planner.py" "planner_curobo.py"
    echo "✓ planner.py -> planner_curobo.py"
else
    echo "✗ planner.py 不存在"
fi

# 2. 重命名 robot.py -> robot_curobo.py
if [ -f "robot.py" ]; then
    mv "robot.py" "robot_curobo.py"
    echo "✓ robot.py -> robot_curobo.py"
else
    echo "✗ robot.py 不存在"
fi

# 3. 重命名 robot_mplib.py -> robot.py
if [ -f "robot_mplib.py" ]; then
    mv "robot_mplib.py" "robot.py"
    echo "✓ robot_mplib.py -> robot.py"
else
    echo "✗ robot_mplib.py 不存在"
fi

# 4. 重命名 planner_mplib.py -> planner.py
if [ -f "planner_mplib.py" ]; then
    mv "planner_mplib.py" "planner.py"
    echo "✓ planner_mplib.py -> planner.py"
else
    echo "✗ planner_mplib.py 不存在"
fi

echo "Now use mplib planner for RL training"        