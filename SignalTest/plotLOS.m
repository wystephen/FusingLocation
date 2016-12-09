clear all;

load('LOS.txt');

% %test heigh
% H = [];
% E = [];
% 
% for heigh = 0.4:0.01:1.5
%     tmp1 = LOS(:,1);
%     tmp2 = (LOS(:,2).^2.0 - heigh ^2.0) .^0.5;
%     
%     use_index = 3;
%     tmp1 = tmp1(1:use_index);
%     tmp2 = tmp2(1:use_index);
%     H = [H;heigh];
%     E = [E;mean(abs(tmp2-tmp1))];
% end
% 
% figure(2);
% 
% plot(H,E,'r+-');
% grid on;    


figure(1);
title('dis to compute dis');
plot(LOS(:,1),(LOS(:,2).^2.0-1.2*1.2).^0.5,'r-+');

X = LOS(:,1);
Y = (LOS(:,2).^2.0-1.2*1.2).^0.5;
grid on;