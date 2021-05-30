function Mtx_without_nans = NanToZero(Mtx)

    index = isnan(Mtx);
    
    Mtx(index) = 0; 
    
    Mtx_without_nans = Mtx;

end