/***read data from external files using data steps****/

Data student;
  infile 'C:\SASCOURSE\data\studenttest.csv';
  Input name $ age grade;
Run;


/***import data from external files****/
PROC IMPORT OUT= WORK.student1
     DATAFILE= "C:\SASCOURSE\data\studenttest.csv"
     DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2;
RUN;
