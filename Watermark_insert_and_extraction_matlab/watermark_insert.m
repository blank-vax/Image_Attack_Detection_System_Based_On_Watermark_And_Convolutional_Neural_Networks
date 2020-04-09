% Function notification
%   This function implements the process of inserting the watermark to original image
%   path1: The path of original image needing the insert of watermark
%   path2: The path of watermark
%   path5: The path of the input image with the insert of watermark

function insert(path1,path2,path5)
im=imread(path1);
mark=imread(path2);
mark=Arnoid(mark,10);
im3=im(:,:,1);
im2=fft2(im3);
flag1=10000;
si=size(im2,1);
re=si/2;
for i=1:128
    for j=1:128
        if(mark(i,j)==1)
            if(real(im2(re+1-i,re+1-j,1))<0)
                im2(re+1-i,re+1-j,1)=real(im2(re+1-i,re+1-j,1))+flag1+abs(real(im2(re+1-i,re+1-j)));
                im2(re+1+i,re+1+j,1)=real(im2(re+1+i,re+1+j,1))+flag1+abs(real(im2(re+1+i,re+1+j)));
            else
                im2(re+1-i,re+1-j,1)=real(im2(re+1-i,re+1-j,1))+flag1;
                im2(re+1+i,re+1+j,1)=real(im2(re+1+i,re+1+j,1))+flag1;
            end
        end
    end
end
ret=ifft2(im2);
im(:,:,1)=ret;
imwrite(im,path5);
end