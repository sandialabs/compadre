x=linspace(-1,1,48);
[X,Y,Z]=meshgrid(x,x,x);
XYZ = [X(:) Y(:) Z(:)] ;
FLAG=zeros(size(XYZ,1),1);
size(FLAG)
for i=1:size(XYZ,1)
    for j=1:3
        if (XYZ(i,j)==1.0 || XYZ(i,j)==-1.0)
            FLAG(i)=1;
        end
    end
end
fid = fopen('flag.txt', 'wt');
for i=1:size(FLAG,1)
    fprintf(fid, '%i\n', FLAG(i));
end
fclose(fid);
fid = fopen('coords.txt', 'wt');
for i=1:size(XYZ,1)
    fprintf(fid, '%f %f %f\n', XYZ(i,1), XYZ(i,2), XYZ(i,3));
end
fclose(fid);
