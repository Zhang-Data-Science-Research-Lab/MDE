betavec = [1:0.5:2]';
Nbeta = size(betavec,1); 
data1 = dlmread('./random_walk_1_90.txt');
[finalY, finalbeta, finalbeta0, y, beta0vec, finalinfo, allY, value] = MDE(betavec, data1);
