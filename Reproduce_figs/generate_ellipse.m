function [x,y]= generate_ellipse(a,b)
x1=a;
x2=-a;
y1=0;
y2=0;
% calculate eccentricity
if a>b
    eccentricity= sqrt(1- b^2/a^2);
else
    eccentricity= sqrt(1- a^2/b^2);
end
numPoints = 100; % Less for a coarser ellipse, more for a finer resolution.
% Make equations:
t = linspace(0, 2 * pi, numPoints); % Absolute angle parameter
X = a * cos(t);
Y = b * sin(t);
% Compute angles relative to (x1, y1).
angles = atan2(y2 - y1, x2 - x1);
x = (x1 + x2) / 2 + X * cos(angles) - Y * sin(angles);
y = (y1 + y2) / 2 + X * sin(angles) + Y * cos(angles);

end