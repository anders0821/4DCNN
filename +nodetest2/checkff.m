function [loss] = checkff(MINIBATCHSIZE, c0P, c1P, cvP, b2P, d5mask, d7mask, f9P, b10P, xO, c0O, c1O, cvO, t3O, s4O, d5O, d6O, flatO, d7O, d8O, b10O, smnO, yO, smlO, offsetId, offset)
    assert(nargin==26);
    assert(any(offsetId==[0 1 2 3 4 5]));
    
    % x
    checkVal = h5read('./data/4D_DATA_FOR_NODE_TEST.mat', '/TESTX', [1 1 1 1 1 1+MINIBATCHSIZE*2], [16 12 3 4 3 MINIBATCHSIZE]);
    if(~isempty(xO))
        fprintf('x\t%e\n', norm(reshape(checkVal-xO,[],1)));
    end
    
    if(offsetId==5)
        checkVal = checkVal+offset;
    end
    
    % conv
    checkLast = checkVal;
    checkVal = zeros([14 10 3 4 2 MINIBATCHSIZE]);% 输出尺寸
    for i=1:MINIBATCHSIZE% N
        for j=1:2% 输出通道
            for k=1:3% 输入通道
                tmp = zeros([size(checkVal,1) size(checkVal,2) size(checkVal,3) size(checkVal,4)]);% 输出尺寸
                for t=1:size(checkVal,4)
                for d=1:size(checkVal,3)
                for w=1:size(checkVal,2)
                for h=1:size(checkVal,1)
                    subA = checkLast(h:h+size(c0P,1)-1, w:w+size(c0P,2)-1, d:d+size(c0P,3)-1, t:t+size(c0P,4)-1, k,i);
                    subK = c0P(:,:,:,:,k,j,h,w,d,t);
                    tmp2 = subA.*subK;
                    tmp(h,w,d,t) = sum(tmp2(:));
                end
                end
                end
                end
                checkVal(:,:,:,:,j,i) = checkVal(:,:,:,:,j,i) + tmp;
            end
        end
    end
    if(~isempty(c0O))
        fprintf('conv\t%e\n', norm(reshape(checkVal-c0O,[],1)));
    end
    
    % conv 手算convn()
    % checkLast = checkVal;
    % EcheckLast = zeros(size(checkLast) + [2 2 2 2 0 0]);
    % EcheckLast(2:end-1,2:end-1,2:end-1,2:end-1,:,:) = checkLast;
    % checkVal = zeros([16 12 3 4 2 MINIBATCHSIZE]);% 输出尺寸
    % for i=1:MINIBATCHSIZE% N
    %     for j=1:2% 输出通道
    %         for k=1:3% 输入通道
    %             for l=1:size(checkLast,4)
    %             for m=1:size(checkLast,3)
    %             for n=1:size(checkLast,2)
    %             for o=1:size(checkLast,1)
    %                 blk = EcheckLast(o:o+2,n:n+2,m:m+2,l:l+2,k,i);
    %                 w = c0P(:,:,:,:,k,j);
    %                 w = flip(w,1);
    %                 w = flip(w,2);
    %                 w = flip(w,3);
    %                 w = flip(w,4);
    %                 checkVal(o,n,m,l,j,i) = checkVal(o,n,m,l,j,i) + sum(sum(sum(sum(blk.*w))));
    %             end
    %             end
    %             end
    %             end
    %         end
    %     end
    % end
    % if(~isempty(c0O))
    %     fprintf('conv\t%e\n', norm(reshape(checkVal-c0O,[],1)));
    % end
    
    if(offsetId==4)
        checkVal = checkVal+offset;
    end
    
    % conv
    checkLast = checkVal;
    checkVal = zeros([12 8 1 2 3 MINIBATCHSIZE]);% 输出尺寸
    for i=1:MINIBATCHSIZE% N
        for j=1:3% 输出通道
            for k=1:2% 输入通道
                tmp = zeros([size(checkVal,1) size(checkVal,2) size(checkVal,3) size(checkVal,4)]);% 输出尺寸
                for t=1:size(checkVal,4)
                for d=1:size(checkVal,3)
                for w=1:size(checkVal,2)
                for h=1:size(checkVal,1)
                    subA = checkLast(h:h+size(c1P,1)-1, w:w+size(c1P,2)-1, d:d+size(c1P,3)-1, t:t+size(c1P,4)-1, k,i);
                    subK = c1P(:,:,:,:,k,j,h,w,d,t);
                    tmp2 = subA.*subK;
                    tmp(h,w,d,t) = sum(tmp2(:));
                end
                end
                end
                end
                checkVal(:,:,:,:,j,i) = checkVal(:,:,:,:,j,i) + tmp;
            end
        end
    end
    if(~isempty(c1O))
        fprintf('conv\t%e\n', norm(reshape(checkVal-c1O,[],1)));
    end

    if(offsetId==3)
        checkVal = checkVal+offset;
    end
    
    % convvalid
    checkLast = checkVal;
    checkVal = zeros([12 8 1 2 4 MINIBATCHSIZE]);% 输出尺寸
    for i=1:MINIBATCHSIZE% N
        for j=1:4% 输出通道
            for k=1:3% 输入通道
                tmp = zeros([size(checkVal,1) size(checkVal,2) size(checkVal,3) size(checkVal,4)]);% 输出尺寸
                for t=1:size(checkVal,4)
                for d=1:size(checkVal,3)
                for w=1:size(checkVal,2)
                for h=1:size(checkVal,1)
                    subA = checkLast(h:h+size(cvP,1)-1, w:w+size(cvP,2)-1, d:d+size(cvP,3)-1, t:t+size(cvP,4)-1, k,i);
                    subK = cvP(:,:,:,:,k,j,h,w,d,t);
                    tmp2 = subA.*subK;
                    tmp(h,w,d,t) = sum(tmp2(:));
                end
                end
                end
                end
                checkVal(:,:,:,:,j,i) = checkVal(:,:,:,:,j,i) + tmp;
            end
        end
    end
    if(~isempty(cvO))
        fprintf('cv\t%e\n', norm(reshape(checkVal-cvO,[],1)));
    end

    % bias
    for i=1:MINIBATCHSIZE% N
        for j=1:4% 输出通道
            checkVal(:,:,:,:,j,i) = checkVal(:,:,:,:,j,i) + b2P(j);
        end
    end

    % LRelu
    checkVal = checkVal.*((checkVal>=0)*(1-0.01)+0.01);
    if(~isempty(t3O))
        fprintf('cbt\t%e\n', norm(reshape(checkVal-t3O,[],1)));
    end
    
    if(offsetId==2)
        checkVal = checkVal+offset;
    end
    
    % maxpool
    sub1 = checkVal(1:2:end, 1:2:end, :, 1:2:end, :, :);
    sub2 = checkVal(2:2:end, 1:2:end, :, 1:2:end, :, :);
    sub3 = checkVal(1:2:end, 2:2:end, :, 1:2:end, :, :);
    sub4 = checkVal(2:2:end, 2:2:end, :, 1:2:end, :, :);
    sub5 = checkVal(1:2:end, 1:2:end, :, 2:2:end, :, :);
    sub6 = checkVal(2:2:end, 1:2:end, :, 2:2:end, :, :);
    sub7 = checkVal(1:2:end, 2:2:end, :, 2:2:end, :, :);
    sub8 = checkVal(2:2:end, 2:2:end, :, 2:2:end, :, :);
    sub1234 = cat(7, sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8);
    checkVal = max(sub1234, [], 7);
    assert(all( size(checkVal)==[6 4 1 1 4 4] ));
    if(~isempty(s4O))
        fprintf('mp\t%e\n', norm(reshape(checkVal-s4O,[],1)));
    end
    
    if(offsetId==1)
        checkVal = checkVal+offset;
    end
    
    % dropoutTrain
    assert(all(all(all(all(all(all( repmat(d5mask(1,1,1,1,:,:), [6 4 1 1 1 1])==d5mask )))))))
    checkVal = checkVal.*d5mask;
    if(~isempty(d5O))
        fprintf('d\t%e\n', norm(reshape(checkVal-d5O,[],1)));
    end
    
    % dropoutTest
    checkVal = checkVal*(round(4*0.7)/4);
    if(~isempty(d6O))
        fprintf('d\t%e\n', norm(reshape(checkVal-d6O,[],1)));
    end
    
    % flatten
    checkVal = reshape(checkVal, [1 1 1 1 6*4*1*1*4 MINIBATCHSIZE]);
    if(~isempty(flatO))
        fprintf('flat\t%e\n', norm(reshape(checkVal-flatO,[],1)));
    end
    
    % dropoutTrain
    checkVal = checkVal .* d7mask;
    if(~isempty(d7O))
        fprintf('d\t%e\n', norm(reshape(checkVal-d7O,[],1)));
    end
    
    % dropoutTest
    checkVal = checkVal*(round(96*0.44)/96);
    if(~isempty(d8O))
        fprintf('d\t%e\n', norm(reshape(checkVal-d8O,[],1)));
    end
    
    % fullcon bias
    checkVal = permute(checkVal, [5 6 1 2 3 4]);
    checkVal = f9P * checkVal;
    checkVal = checkVal + repmat(b10P, [1 MINIBATCHSIZE]);
    checkVal = permute(checkVal, [3 4 5 6 1 2]);
    if(~isempty(b10O))
        fprintf('fc\t%e\n', norm(reshape(checkVal-b10O,[],1)));
    end
    
    % smn
    checkVal = permute(checkVal, [5 6 1 2 3 4]);
    for i=1:MINIBATCHSIZE
        prob = checkVal(:,i);
        prob = exp(prob);
        prob = prob / sum(prob);
        checkVal(:,i) = prob;
    end
    checkVal = permute(checkVal, [3 4 5 6 1 2]);
    if(~isempty(smnO))
        fprintf('smn\t%e\n', norm(reshape(checkVal-smnO,[],1)));
    end
    
    % y
    checkVal2 = h5read('./data/4D_DATA_FOR_NODE_TEST.mat', '/TESTY', [1 1 1 1 1 1+MINIBATCHSIZE*2], [1 1 1 1 7 MINIBATCHSIZE]);
    if(~isempty(yO))
        fprintf('y\t%e\n', norm(reshape(checkVal2-yO,[],1)));
    end
    
    % sml
    lbl = permute(checkVal2, [5 6 1 2 3 4]);
    [~, lbl] = max(lbl);
    idx = sub2ind([1 1 1 1 7 MINIBATCHSIZE], ones(1,MINIBATCHSIZE), ones(1,MINIBATCHSIZE), ones(1,MINIBATCHSIZE), ones(1,MINIBATCHSIZE), lbl, 1:MINIBATCHSIZE);
    checkVal = checkVal(idx);
    checkVal = -log(checkVal);
    checkVal = mean(checkVal);
    if(~isempty(smlO))
        fprintf('sml\t%e\n', checkVal-smlO);
    end
    
    loss = checkVal;
end
