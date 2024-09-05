#!/bin/bash

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path_to_notebook"
    exit 1
fi

# Assign the first argument to the variable notebook
notebook=$1

# Now you can use the variable $notebook in your script
echo "The path to the notebook is \"$notebook\""

# Extract, modify, and save the pip install command
extract_and_modify_pip_installation() {
    echo "Extracting and processing code cells from notebook..."

    # Clear or create the additional_requirements.txt file
    > additional_requirements.txt

    # 1. Extract code cells from the notebook using jq and strip surrounding quotes
    jq '.cells[] | select(.cell_type == "code") | .source[]' "$1" | sed 's/^"//;s/"$//' | \
    while read -r line ; do
        # Check for pip install lines
        if echo "$line" | grep -E '^!pip(3)? install( -U)? ' > /dev/null; then
            echo "Found pip install line: $line"

            # Strip the leading "!" and remove 'pip(3) install' and '-U' flags
            modified_line=$(echo "$line" | sed 's/^!pip[3]* install -U //;s/^!pip[3]* install //')

            # Remove 'pyvespa' and 'vespacli' from the line
            modified_line=$(echo "$modified_line" | sed 's/pyvespa//g' | sed 's/vespacli//g' | sed 's/  / /g')

            # Write each package to additional_requirements.txt without adding extra new lines
            echo "$modified_line" | tr ' ' '\n' | sed '/^$/d' >> additional_requirements.txt

            # Echo the modified command for verification
            echo "Modified command for requirements.txt: $modified_line"

            # Comment out the original line in the notebook by prefixing it with '#'
            tmpfile=$(mktemp)
            jq --arg old "$line" --arg new "# $line" '
                (.cells[] | select(.cell_type == "code") | .source[] | select(. == $old)) = $new
            ' "$1" > "$tmpfile" && mv "$tmpfile" "$1"
        fi
    done

    echo "Finished processing."
}

# Call the function to process pip install lines and modify the notebook
extract_and_modify_pip_installation "$notebook"