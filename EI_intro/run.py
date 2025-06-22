import os
import sys
import subprocess

def run_script(script_name):
    try:
        result = subprocess.run(['python', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        exit(1)

def main():
    print("Training...")
    run_script(r'train/train.py')

    choice = input("Run demo? [Y/n]: ").strip().lower()
    if choice == 'y' or choice == '':
        print("Running demo ...")
        run_script(r'demo/demo.py')
    else:
        print("Exited。")

if __name__ == "__main__":
    main()
