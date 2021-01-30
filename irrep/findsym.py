import subprocess

fsout=subprocess.run(['./findsym','findsym_sample.in'],capture_output=True)
fsout=fsout.stdout.decode('utf-8').split('\n')