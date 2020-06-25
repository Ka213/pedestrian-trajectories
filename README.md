# Master Thesis: Long-Term Motion prediction in traffic

### Installation:

Satisfy all the requirements in requirenments.txt:

    pip install -r requirements.txt
   
The pyrieef project should now be inside a folder named src.

Execute file:

    python learch.py

If you execute the file learch.py the first time, 
the module 'common_imports' in src/pyrieef/pyrieef/learning/inverse_optimal_control.py can occur.
Change the 21th line to 'import learning.common_imports'.

Another TypeError in line 48 of the the same file can occur. 
Change it to 'InverseOptimalControl.__init__(self,nb_demonstrations)'.