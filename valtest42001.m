clc;
clear;
close all;
drawnow;
rng default;

for RESUME_FROM_SNAPSHOT_ITER = []
    util.dispspliter(num2str(RESUME_FROM_SNAPSHOT_ITER))
    for SUBSET = {'TEST', 'VAL', 'TRAIN'}
        MINIBATCHSIZE = 12;
        
% 初始化网络
x = node.Data('./data/XY3_ALLFIDS_NOBG_INTERP_MIRROR.mat', [SUBSET{1} 'X'], MINIBATCHSIZE);
y = node.Data('./data/XY3_ALLFIDS_NOBG_INTERP_MIRROR.mat', [SUBSET{1} 'Y'], MINIBATCHSIZE);
xg = node.OpticalFlow(x, 'same');
xc = node.Crop(xg, [108 108 16 1], 'center');

c11 = node.ConvValidParallelStopBp(xc, [3 3 3 1], 10);       c11.P = randn(size(c11.P)) * 0.01*450;
b11 = node.Bias(c11);
t11 = node.LRelu(b11);
c12 = node.ConvValidParallel(t11, [3 3 1 1], 10);      c12.P = randn(size(c12.P)) * 0.01*20;
b12 = node.Bias(c12);
t12 = node.LRelu(b12);
c13 = node.ConvValidParallel(t12, [3 3 3 1], 10);      c13.P = randn(size(c13.P)) * 0.01*9;
b13 = node.Bias(c13);
t13 = node.LRelu(b13);
c14 = node.ConvValidParallel(t13, [3 3 1 1], 10);      c14.P = randn(size(c14.P)) * 0.01*16;
b14 = node.Bias(c14);
t14 = node.LRelu(b14);
s1 = node.MaxPool(t14, [1 1 1 0]);
d1 = node.DropoutTest(s1,0.5);

c21 = node.ConvValidParallel(d1, [3 3 3 1], 12);      c21.P = randn(size(c21.P)) * 0.01*13;
b21 = node.Bias(c21);
t21 = node.LRelu(b21);
c22 = node.ConvValidParallel(t21, [3 3 3 1], 12);     c22.P = randn(size(c22.P)) * 0.01*8;
b22 = node.Bias(c22);
t22 = node.LRelu(b22);
s2 = node.MaxPool(t22, [1 1 1 0]);
d2 = node.DropoutTest(s2,0.5);

flat = node.Flatten(d2);

% 最后一个全连接层 表示prob 不用tanh归一化用SoftMax归一化 即SoftMaxNorm(W*x+b)
f7 = node.FullCon(flat, 6);       f7.P = randn(size(f7.P)) * 0.01;
b7 = node.Bias(f7);
smn = node.SoftMaxNorm(b7);

% 交叉熵损失 输入prob与label计算标量损失
sml = node.SoftMaxLoss(smn, y);

        
        % 如果需要则从快照中恢复
        if(RESUME_FROM_SNAPSHOT_ITER>0)
            fn = ['snapshot/snapshot-' num2str(RESUME_FROM_SNAPSHOT_ITER) '.mat'];
            load(fn, 'params');
            sml.setPs(params);
        end
        
        % 主迭代 计算confusePair
        MAX_ITER = ceil( x.dim(6)/MINIBATCHSIZE );
        confusePair = zeros(2, MINIBATCHSIZE, MAX_ITER);
        util.dispstat('', 'init');
        fprintf('%s %d\n', SUBSET{1}, x.dim(6))
        for i=1:MAX_ITER
            % ff
            sml.ff();
            
            % 写confusePair
            [~,lbl] = max(permute(smn.O, [5 6 1 2 3 4]));
            [~, groundtruthLbl] = max(permute(y.O, [5 6 1 2 3 4]));
            confusePair(1,:,i) = groundtruthLbl;
            confusePair(2,:,i) = lbl;
            
            % % 可视化
            % for j=1:MINIBATCHSIZE
            %     subplot(1,MINIBATCHSIZE,j);
            %     imshow(x.O(:,:,1,1,1,j));
            %     title([num2str(groundtruthLbl(j)) ' -> ' num2str(lbl(j))])
            %     drawnow();
            % end
            
            curAcc = sum(sum(confusePair(1,:,1:i)==confusePair(2,:,1:i),2),3) / i / MINIBATCHSIZE;
            util.dispstat(sprintf('Processing %d%% %d/%d %f%%', round(i/MAX_ITER*100), i*MINIBATCHSIZE, MAX_ITER*MINIBATCHSIZE, round(curAcc*100)));
        end
        util.dispstat(' ');
        
        % 截断最后一个超出的minibatch
        confusePair = reshape(confusePair, [2 MINIBATCHSIZE*MAX_ITER]);
        confusePair = confusePair(:, 1:x.dim(6));
        
        % 计算acc
        acc = sum(confusePair(1,:)==confusePair(2,:)) / size(confusePair,2);
        disp(acc);
        
        % 计算混淆矩阵
        [M] = confusionmat(confusePair(1,:), confusePair(2,:), 'order', 1:6);
        R = round(diag(M)./sum(M,2) * 100);
        B = round(diag(M)'./sum(M,1) * 100);
        BR = round(sum(diag(M)) / sum(sum(M)) * 100);
        disp(num2str([M R
            B BR]));
    end
end

%util.beep(100);
