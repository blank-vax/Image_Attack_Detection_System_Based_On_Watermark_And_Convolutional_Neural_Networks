% This part defines the implementations details of the arnold substitute

function arnoldImg = Arnoid(img,n)
a=1;
b=1;
[h,w] = size(img);
N=h;
arnoldImg = zeros(h,w); 
for i=1:n
    for y=1:h
        for x=1:w  
            xx=mod((x-1)+b*(y-1),N)+1; 
            yy=mod(a*(x-1)+(a*b+1)*(y-1),N)+1; 
            arnoldImg(yy,xx)=img(y,x);
        end
    end
    img=arnoldImg; 
end
end
