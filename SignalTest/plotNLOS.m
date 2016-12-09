clear all;
load('NLOS.txt');


z_offset = 1.20;% 

x_offset = 4 * 0.8 + 0.15;

y_offset = 3 * 0.8 - 0.15;

real_dis = zeros(size(NLOS));
index_x = zeros(size(NLOS));
index_y = zeros(size(NLOS));

for i = 1:size(NLOS,1)
    for j = 1:size(NLOS,2)
        index_x(i,j) = j;
        index_y(i,j) = i;
        real_dis(i,j) = (((i-1) * 0.8 + y_offset)^2.0 + ...
            ((j-1) * 0.8 + x_offset)^2.0 + ...
            z_offset^2.0)^0.5
    end
end

figure(1);
%surf(NLOS,'r');
%surf(real_dis,'b');
%mesh((NLOS-real_dis));
contour((NLOS-real_dis));

X = NLOS(:);
Y = real_dis(:);

IX = index_x(:);
IY = index_y(:);
error = abs(NLOS-real_dis);
IZ1 = error(:);

