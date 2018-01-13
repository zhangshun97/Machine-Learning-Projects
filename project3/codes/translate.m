function r = translate(X,a,b)
% this function aims to translate a matrix
% a acounts for the step down(a>0) and up(a<0)
% b acounts for the step right(a>0) and left(a<0)
% X is defaulted as an array of size (1,256)

f = reshape(X,16,16);
[m,n] = size(f);
move_step = [a b];
abs_move_step = abs(move_step);
g = padarray(f,abs_move_step,0);
r = circshift(g,move_step);
r = r(abs_move_step(1)+1:abs_move_step(1)+m,abs_move_step(2)+1:abs_move_step(2)+n);