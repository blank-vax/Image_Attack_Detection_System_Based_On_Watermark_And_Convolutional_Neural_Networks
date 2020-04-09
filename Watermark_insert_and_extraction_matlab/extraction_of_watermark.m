% Function notification:
%   This function implements the extraction of watermark from the processed image
%   path3 is the path of processed image
%   path4 is the path of generated watermark

function extract(path3,path4) 
im=imread(path3);
ret=im(:,:,1);
q=zeros(128,128);
qq=fft2(ret);
for i=1:128
    for j=1:128
        % Get rid of the outliers 
        if(real(qq(321-i,321-j))>2000)
            q(i,j)=1;
        end
    end
end
q=rearnold(q,10);
imwrite(q,path4);
end