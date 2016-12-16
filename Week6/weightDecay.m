inData = csvread('inData.csv');
outData = csvread('outData.csv');

inDataTransform = zeros(35, 8);
yMatrixIn = zeros(35, 1);

for i=1:35;
	inLine = inData(i, :);
	x1 = inLine(1);
	x2 = inLine(2);
	inDataTransform(i, :) = [1, x1, x2, x1^2, x2^2, x1*x2, abs(x1-x2), abs(x1+x2)];
	yMatrixIn(i) = inLine(3);
end

k = -1;
lambda = 10^k;
innerResult = (transpose(inDataTransform) * inDataTransform) + lambda*eye(8);
weight = inv(innerResult) * transpose(inDataTransform) * yMatrixIn;

Ein = 0;
for i=1:35;
	result = inDataTransform(i, :) * weight;
	if(sign(result) ~= yMatrixIn(i))
		Ein = Ein + 1;
	end
end


Eout = 0;
for i=1:250;
	outLine = outData(i, :);
	x1 = outLine(1);
	x2 = outLine(2);
	yResult = outLine(3);
	outDataTransform = [1, x1, x2, x1^2, x2^2, x1*x2, abs(x1-x2), abs(x1+x2)];
	result = outDataTransform * weight;
	if(sign(result) ~= yResult)
		Eout = Eout + 1;
	end
end

EoutError = Eout/250
EinError = Ein/35
