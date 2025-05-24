# Python details Compute Canada
1. module avail python : list available versions of python
 ipython-kernel/3.10              
 ipython-kernel/3.11    python/3.10.13            
 ipython-kernel/3.12    python/3.11.5 
 ipython-kernel/3.13     python/3.12.4
2.  load the version of your choice
`module load python/3.12`
3.  scipy-stack module includes: NumPy, SciPy, Matplotlib, IPython
`module load scipy-stack`
4. using a virtual environment
create in home directory/[inside jobs](https://docs.alliancecan.ca/wiki/Python#Creating_virtual_environments_inside_of_your_jobs)
    - To create in home directory:
    ```
    cd ~
    python -m venv ~/.venvs/venv
    cd <project_folder>
    source ~/.venvs/venv/bin/activate
    pip install --no-index --upgrade pip
    pip install torch --no-index #install required packages
    pip install -r requirements.txt --no-index
    deactivate #to exit venv
    ```
5. Register the kernel:
`python -m ipykernel install --user --name venv --display-name "Python (venv)"`
6. You can now use the same virtual environment over and over again. Each time:

Load the same environment modules that you loaded when you created the virtual environment, e.g. `module load python scipy-stack`
Activate the environment, `source ~/.venvs/venv/bin/activate`

7. Allot interactive job with GPU (compute node)
`salloc --time=1:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=16G --account=def-mushrifs  `
note: compute nodes don't have internet access. you will have to install missing python packages to venv, data or pretained model etc from a login node.


### Relevant links
https://docs.alliancecan.ca/wiki/Tutoriel_Apprentissage_machine/en

