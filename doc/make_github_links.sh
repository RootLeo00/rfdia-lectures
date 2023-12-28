#!/bin/bash

dir="$PWD"

# Iterate through every file in every subfolder of $PWD
find "$PWD" -type f -name "*.md" | while read -r file
do
    echo "$file";
    echo "Processing: $file";
    sed -ri 's|(\!\[\[Pasted\ image\ )|![300](./images/Pasted%20image%20|g' "$file";
    sed -ri 's|(\]\])|)|g' "$file";
done;

# Consider images under /images folder.
# old image link   -  ![[Pasted image 20220109203352.png]] 
# new image link - ![](./images/Pasted%20image%2020220122124535.png) 
