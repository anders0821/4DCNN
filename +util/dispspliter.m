function [] = dispspliter(msg)
    msg = [' ' msg ' '];
    i = size(msg,2);
    pad = 80-i;
    if(pad<0)
        pad=0;
    end
    lpad = floor(pad/2);
    rpad = ceil(pad/2);
    assert(lpad+rpad==pad);
    disp([ones([1, lpad])*'-' msg ones([1, rpad])*'-']);
end
