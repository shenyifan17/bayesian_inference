function [Y] = Sigmoid(X)
        
    Y = 1./(exp(-X) + 1);

end