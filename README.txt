name: Saurabh Mylavaram
student ID: 5593072
email ID: mylav008@umn.edu
----------------------------------

HOW TO RUN THE CODE
-------------------
Put the data file ('breast-cancer-wisconsin.data') inside 
a folder named 'data' in the current directory, and then run the following:
1. 'python adaboost.py'
2. 'python rf.py'

Put the image 'umn-csci.png' in the same directory and run
3. 'python kmeans.py'

ASSUMPTIONS
-------------
1. For problem 1 and 2(i), we ignored the first column (patient IDs) as they are irrelevant to predicting cancer.
2. However, for problem 2(ii) we included this column.
3. Replace the missing values in data by the corresponding feature's mode value.

4. For easy comparison, I'm printing error-rates after steps of adding 10 learners (like 10,20,30... learners).
   We can change this to print for every number between 1-100 (like 1,2,3,... learners) easily.
