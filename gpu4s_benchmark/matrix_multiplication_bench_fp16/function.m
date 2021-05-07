N=256;
a=rand(N);
b=inv(a);
c=a*b;
max(max(abs(c-eye(N))))
fileID = fopen('matX256in.bin','w');
fwrite(fileID, N ,  'float64', 0, 'l')
fwrite(fileID, a' , 'float64', 0, 'l')
fwrite(fileID, b' , 'float64', 0, 'l')
fclose(fileID);




fileID = fopen('matX256in.bin','r');
N = fread(fileID, [1 1] ,  'float64', 0, 'l')
a = fread(fileID, [N N] ,  'float64', 0, 'l')';
b = fread(fileID, [N N] ,  'float64', 0, 'l')';
fclose(fileID);


d = a*b;
max(max(abs(c-d)))

