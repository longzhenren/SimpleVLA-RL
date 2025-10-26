#!/bin/bash

# Check number of arguments
if [ "$#" -ne 2 ]; then
    echo "Error: Two arguments required"
    echo "Usage: $0 <robotwin_path> <simplevlarl_path>"
    exit 1
fi

# Get arguments
robotwin_path="$1"
simplevlarl_path="$2"

# Target path
target_path="${simplevlarl_path}/verl/utils/envs/robotwin2"

# Modified codes path
modified_path="${simplevlarl_path}/modified_codes/robotwin2"

# Check if source path exists
if [ ! -d "$robotwin_path" ]; then
    echo "Error: Source path '$robotwin_path' does not exist or is not a directory"
    exit 1
fi

# Check if simplevlarl path exists
if [ ! -d "$simplevlarl_path" ]; then
    echo "Error: simplevlarl path '$simplevlarl_path' does not exist or is not a directory"
    exit 1
fi

# Create target directory if it doesn't exist
echo "Creating target directory (if not exists): $target_path"
mkdir -p "$target_path"

# Step 1: Execute initial copy operation
echo "Step 1: Copying files from robotwin_path..."
echo "From: $robotwin_path"
echo "To: $target_path"

# Use cp -r to copy all content, -f to force overwrite
cp -rf "$robotwin_path"/* "$target_path/" 2>/dev/null

# Check if copy was successful
if [ $? -eq 0 ]; then
    echo "Initial copy successful!"
else
    # If source folder is empty, cp will return an error, we need to check this case
    if [ -z "$(ls -A "$robotwin_path")" ]; then
        echo "Warning: Source folder is empty, no files were copied"
    else
        echo "Error occurred during copy operation"
        exit 1
    fi
fi

# Display count of copied files
file_count=$(find "$target_path" -type f | wc -l)
dir_count=$(find "$target_path" -type d | wc -l)
echo "Copied $file_count files and $dir_count directories to target path"

# Step 2: Overwrite with modified files
echo ""
echo "Step 2: Overwriting with modified files..."

# Check if modified_codes directory exists
if [ ! -d "$modified_path" ]; then
    echo "Warning: Modified codes directory '$modified_path' does not exist"
    echo "Skipping overwrite step..."
else
    echo "From: $modified_path"
    echo "To: $target_path"
    
    # Count files to be overwritten
    overwrite_count=0
    
    # Find all files in modified_path and copy them to corresponding locations in target_path
    while IFS= read -r -d '' file; do
        # Get relative path from modified_path
        relative_path="${file#$modified_path/}"
        
        # Construct target file path
        target_file="$target_path/$relative_path"
        
        # Create target directory if it doesn't exist
        target_dir=$(dirname "$target_file")
        mkdir -p "$target_dir"
        
        # Copy the file with overwrite
        echo "Overwriting: $relative_path"
        cp -f "$file" "$target_file"
        
        if [ $? -eq 0 ]; then
            ((overwrite_count++))
        else
            echo "Error: Failed to overwrite $relative_path"
        fi
    done < <(find "$modified_path" -type f -print0)
    
    if [ $overwrite_count -gt 0 ]; then
        echo "Successfully overwritten $overwrite_count file(s)"
    else
        echo "No files were overwritten"
    fi
fi

echo ""
echo "Step 3: Running update_embodiment_config_path.py..."

# Construct the script path
update_script="${target_path}/script/update_embodiment_config_path.py"

# Change to target directory
cd "$target_path"

# Check if the script exists
if [ ! -f "$update_script" ]; then
    echo "Error: Script '$update_script' does not exist"
    echo "Skipping script execution..."
else
    echo "Executing: python $update_script"
    python "$update_script"
    
    if [ $? -eq 0 ]; then
        echo "Script executed successfully!"
    else
        echo "Error: Script execution failed"
        exit 1
    fi
fi

# Step 4: Remove pencil folder
echo ""
echo "Step 4: Removing pencil folder..."

pencil_path="${target_path}/assets/objects/objaverse/pencil"

# Check if pencil folder exists
if [ -d "$pencil_path" ]; then
    echo "Found pencil folder at: $pencil_path"
    echo "Removing folder..."
    
    # Remove the folder and its contents
    rm -rf "$pencil_path"
    
    if [ $? -eq 0 ]; then
        echo "Pencil folder successfully removed!"
    else
        echo "Error: Failed to remove pencil folder"
        exit 1
    fi
else
    echo "Pencil folder not found at: $pencil_path"
    echo "Nothing to remove, continuing..."
fi

echo ""
echo "All operations completed!"