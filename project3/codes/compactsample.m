function Xout = compactsample(Xin)
% Xin is supposed to have size (1,257)

Xt = reshape(Xin(2:end),16,16);

% X will be translated slightly by random steps from 0 to 1
% also random with up/down/left/right
a = randi(3,1)-2;
b = randi(3,1)-2;
Xt = translate(Xt,a,b);

% X will be rotated slightly by random degrees from 0 to 2
% also random with clockwise or anti-clockwise
theta = rand(1)*4 - 2;
Xt = imrotate(Xt, theta);
ss = size(Xt);
Xt = imresize(Xt,16/ss(1));
Xout = reshape(Xt,1,256);
Xout = [1 Xout];
% the output sample maintains the size (1,257)