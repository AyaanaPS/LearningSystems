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

k = 7;
validation = inDataTransform(1:25, 1:k+1);
training = inDataTransform(26:35, 1:k+1);

pseudoPoints = pinv(training);
weight = pseudoPoints * yMatrixIn(26:35);

Eval = 0;
for i=1:25;
	result = validation(i,:) * weight;
	if(sign(result) ~= yMatrixIn(i))
		Eval = Eval + 1;
	end
end

Eout = 0;
for i=1:250;
	outLine = outData(i, :);
	x1 = outLine(1);
	x2 = outLine(2);
	yResult = outLine(3);
	outDataTransform = [1, x1, x2, x1^2, x2^2, x1*x2, abs(x1-x2), abs(x1+x2)];
	result = outDataTransform(1:k+1) * weight;
	if(sign(result) ~= yResult)
		Eout = Eout + 1;
	end
end

EvalError = Eval/25
EoutError = Eout/250
