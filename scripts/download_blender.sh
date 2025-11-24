#!/bin/bash

cd ./objaverse_eval/

BLENDER_VERSION="3.3.21"
BLENDER_URL="https://download.blender.org/release/Blender3.3/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
ARCHIVE_NAME="blender-${BLENDER_VERSION}-linux-x64.tar.xz"

echo "Downloading Blender ${BLENDER_VERSION}..."
wget -c $BLENDER_URL

if [ -f "$ARCHIVE_NAME" ]; then
    echo "Extracting..."
    tar -xf $ARCHIVE_NAME
    
    echo "Cleaning up..."
    rm $ARCHIVE_NAME
    
    echo "Done!"
else
    echo "Download failed."
    exit 1
fi