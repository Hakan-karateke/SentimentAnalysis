#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to copy the text preprocessing module to the ensemble model directory
"""
import os
import shutil

def update_preprocessing_file():
    source_file = "sentiment_model/text_preprocessing.py"
    target_file = "models/ensemble_sentiment_model/text_preprocessing.py"
    
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return False
    
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        # Copy the file
        shutil.copy2(source_file, target_file)
        print(f"Successfully copied {source_file} to {target_file}")
        return True
    except Exception as e:
        print(f"Error copying file: {str(e)}")
        return False

if __name__ == "__main__":
    update_preprocessing_file()
