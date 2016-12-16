% Number of repetitions for the simulation
repetitions = 100000;
runs = 0;

% Gathering data to compute the average after all the repetitions.
sumFirst = 0;
sumRand = 0;
sumMin = 0;

while (runs < repetitions)
    
    numCoins = 1000;
    
    coin = 1;
    
    % Choosing a random coin in the 1000 to focus on.
    randCoin = randi(1000, 1);
    firstHeads = 0;
    randHeads = 0;
    minHeads = 0;
    
    % Flipping each of the 1000 coins 10 times and counting the heads.
    while(coin <= numCoins)
        
        flips = 0;
        numHeads = 0;
        
        while(flips < 10)
            
            side = randi(2,1);
            % Heads = 1, Tails = 2
            if(side == 1)
                numHeads = numHeads + 1;
            end
            
            flips = flips + 1;
        end
        
        % Stores the data if the coin is the first coin or the random one.
        if(coin == 1)
            firstHeads = numHeads;
        elseif(coin == randCoin)
            randHeads = numHeads;
        end
        
        % Checks to see if a new minimum has been found.
        if(numHeads < minHeads)
            minHeads = numHeads;
        end
        
        coin = coin + 1;
    end
    
    runs = runs + 1;
    sumFirst = sumFirst + firstHeads;
    sumRand = sumRand + randHeads;
    sumMin = sumMin + minHeads;
    
end

% Computing the average number of heads
numFirst = sumFirst/repetitions;
numRand = sumRand/repetitions;
numMin = sumMin/repetitions;

% Printing out each of the fractions
fracFirst = numFirst/10
fracRand = numRand/10
fracMin = numMin/10
