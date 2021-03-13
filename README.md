
### Trouble Shooting

for mac user:

Q: 
Cannot compile MPI programs. Check your configuration!!!

A: 
run `brew install mpich` before install requirements

Q:
how to get rom

A: 
https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed

A:
No such file or directory: 'ffmpeg': 'ffmpeg'

Q:
`brew install ffmpeg`

A:
Error occur when use SubprocVecEnv:
```
An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
```

Q:
must train wrapped in 
```
if __name__ == '__main__':
```