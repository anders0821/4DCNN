function [] = beep(N)
    for i=1:N
        load train
        sound(y,Fs)
        pause(3)
    end
end
