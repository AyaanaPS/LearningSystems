% Sets the number of random points
N = 10;

numRuns = 0;

SVMbetter = 0;
avgSupport = 0;

while (numRuns < 1000)

    % Creates the dataset of random points
    points = -1 + (2)*rand(N, 3);
    for i=1:N;
        points(i,1) = 1;
    end

    % Creates the two points that make up the lines
    A = -1+(2)*rand(1,2);
    B = -1+(2)*rand(1,2);

    % Initializes the solution matrix
    yMatrix = zeros(N,1);

    % Fills the solution matrix by calculating the determinant
    for i=1:N;
        point = points(i,:);
        value = (B(1)-A(1))*(point(3)-A(2))-(B(2)-A(2))*(point(2)-A(1));
        pos = sign(value);
        yMatrix(i) = pos;
    end

    % Initializes the weight matrix
    weight = zeros(1,3);

    % Initially, the set of misclassified points is all the points
    misClass = points;
    % Similarily the solution matrix for misclassified points has all the
    % solutions.
    yVals = yMatrix;
    % This is a counter for the number of times we update the weight
    numUpdates = 0;

    % We continue to iterate until there are no more misclassified points
    while(isempty(misClass) ~= 1)
    
        % First we choose a random misclassified point
        temp = size(misClass);
        numMiss = temp(1);
        pointN = randi([1 numMiss], 1, 1);
        pointChosen = misClass(pointN, :);
  
        % Double check that it is indeed misclassified
        transposedWeight = transpose(weight);
        matrixResult = pointChosen * transposedWeight;
        if(sign(matrixResult) ~= yVals(pointN))
            numUpdates = numUpdates + 1;
            weight = weight + yVals(pointN)*pointChosen;
        end
    
        % Recompute the misclassified points
        newMisClass = [];
        newYVals = [];
    
        for i=1:N;
            point = points(i,:);
            transposedWeight2 = transpose(weight);
            matrixResult2 = point * transposedWeight2;
            if(sign(matrixResult2) ~= yMatrix(i))
                newMisClass = [newMisClass ; point];
                newYVals = [newYVals ; yMatrix(i)];
            end
        end
    
        misClass = newMisClass;
        yVals = newYVals;
    end

    H = zeros(N, N);

    for i=1:N;
   	    for k=1:N;
    	   pointI = points(i, 2:3);
    	   curPoint = transpose(points(k, 2:3));
           H(i,k) = yMatrix(i) * yMatrix(k) * pointI * curPoint;
        end
    end

    F = ones(N, 1);
    F = -1*F;

    options = optimoptions('quadprog', 'Display', 'None');
    alpha = quadprog(H, F, transpose(yMatrix), 0, transpose(yMatrix), 0, zeros(N,1), Inf(N,1), [], options);
    
    weightSVM = 0;
    numSupport = 0;
    for i=1:N;
        point = points(i,:);
        val = alpha(i) * yMatrix(i) * point;
        if(alpha(i) > 1)
            numSupport = numSupport + 1;
        end
        weightSVM = weightSVM + val;
    end
        
    avgSupport = avgSupport + numSupport;

    testPoints = -1 + (2)*rand(500, 3);
    SVMcount = 0;
    PLAcount = 0;
    for i=1:500;
        point = testPoints(i, :);
        value = (B(1)-A(1))*(point(3)-A(2))-(B(2)-A(2))*(point(2)-A(1));
        position = sign(value);
        pointMatrixPLA = point * transpose(weight);
        pointMatrixSVM = point * transpose(weightSVM);
        if(sign(pointMatrixPLA) ~= position)
            PLAcount = PLAcount + 1;
        end
        if(sign(pointMatrixSVM) ~= position)
            SVMcount = SVMcount + 1;
        end
    end

    if(SVMcount < PLAcount)
        SVMbetter = SVMbetter + 1;
    end

    numRuns = numRuns + 1;
end

finalSVM = SVMbetter/(1000)
finalSupport = avgSupport/(1000)
