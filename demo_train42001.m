clc;
clear;
close all;
drawnow;
rng shuffle;

% 算法超参数
% 参考AlexNet: MB 128 LR 0.01...0.001...0.0001...0.00001 WD 0.0005 MOM 0.9
% SGD：(MB越大CPU利用率越高 尤其不要小于CPU线程数 但MB越大内存消耗越大)
%      (MB*BB为经典定义的mini-batch大小 由于随机采样受噪声影响不再下降 则提高该值有助于减小误差)
%      (MB*BB积相等时，因子的不同组合不影响计算(浮点误差和Crop+DropoutTrain随机数发送顺序会影响结果)，如64*1 8*8两种配置的计算结果是一致的)
% 例子：起始MB=CPU线程数 BB=1
%      不下降后MB*=3
%      若内存满则MB不变BB*=3
MINIBATCHSIZE = 4;
BIGBATCHSIZE = 1;% 1 3 7 16
LR = 0.01 * sqrt(MINIBATCHSIZE*BIGBATCHSIZE/128);
WD = 0.0;
MOM = 0.9;

% 程序参数
DISPLAY_LOSSES_ACCS_INTERVAL = ceil(100/MINIBATCHSIZE/BIGBATCHSIZE);
DISPLAY_HIST_INTERVAL = 1e12;
SNAPSHOT_INTERVAL = ceil(100/MINIBATCHSIZE/BIGBATCHSIZE);
RESUME_FROM_SNAPSHOT_ITER = 0;
MAX_ITER = 1e12;

% 初始化网络
x = node.Data('./data/XY3_ALLFIDS_NOBG_INTERP_MIRROR.mat', 'TRAINX', MINIBATCHSIZE);
y = node.Data('./data/XY3_ALLFIDS_NOBG_INTERP_MIRROR.mat', 'TRAINY', MINIBATCHSIZE);
xg = node.OpticalFlow(x, 'same');
xc = node.Crop(xg, [108 108 16 1], 'rand');

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
d1 = node.DropoutTrain(s1,0.5);

c21 = node.ConvValidParallel(d1, [3 3 3 1], 12);      c21.P = randn(size(c21.P)) * 0.01*13;
b21 = node.Bias(c21);
t21 = node.LRelu(b21);
c22 = node.ConvValidParallel(t21, [3 3 3 1], 12);     c22.P = randn(size(c22.P)) * 0.01*8;
b22 = node.Bias(c22);
t22 = node.LRelu(b22);
s2 = node.MaxPool(t22, [1 1 1 0]);
d2 = node.DropoutTrain(s2,0.5);

flat = node.Flatten(d2);

% 最后一个全连接层 表示prob 不用tanh归一化用SoftMax归一化 即SoftMaxNorm(W*x+b)
f7 = node.FullCon(flat, 6);       f7.P = randn(size(f7.P)) * 0.01;
b7 = node.Bias(f7);
smn = node.SoftMaxNorm(b7);

% 交叉熵损失 输入prob与label计算标量损失
sml = node.SoftMaxLoss(smn, y);

% debugMemory
[totalMemory, totalParamMemory] = sml.debugMemory();
totalMemory = totalMemory*8/1024/1024/1204;
totalParamMemory = totalParamMemory*8/1024/1024/1204;
fprintf('totalMemory: %f GB\ntotalParamMemory: %f GB\n', totalMemory, totalParamMemory);

% % 剖析ff bp时间
% disp('------------------------------------------------------------------------------------------------')
% profile on;
% tic;
% for i=1:10
%     sml.ff();
%     sml.bp();
% end
% toc;
% profile off;
% profile viewer;
% return;

disp('------------------------------------------------------------------------------------------------')
% 学习过程中solver管理的状态量
losses = cell(0);
accs = cell(0);
v = cell(0);
v{1} =  zeros(size(c11.gradP));
v{2} =  zeros(size(b11.gradP));
v{3} =  zeros(size(c12.gradP));
v{4} =  zeros(size(b12.gradP));
v{5} =  zeros(size(c13.gradP));
v{6} =  zeros(size(b13.gradP));
v{7} =  zeros(size(c14.gradP));
v{8} =  zeros(size(b14.gradP));
v{9} =  zeros(size(c21.gradP));
v{10} =  zeros(size(b21.gradP));
v{11} =  zeros(size(c22.gradP));
v{12} =  zeros(size(b22.gradP));
v{13} =  zeros(size( f7.gradP));
v{14} =  zeros(size( b7.gradP));

% 如果需要则从快照中恢复
if(RESUME_FROM_SNAPSHOT_ITER>0)
    fn = ['snapshot/snapshot-' num2str(RESUME_FROM_SNAPSHOT_ITER) '.mat'];
    load(fn, 'params', 'losses', 'accs', 'v');
    fprintf('load from %s\n', fn);
    sml.setPs(params);
end

% 主迭代
for iter=(RESUME_FROM_SNAPSHOT_ITER+1):MAX_ITER
    t = tic;
    fprintf('iter: %d\n', iter);
    
    % BIG BATTCH累加到gradPAvg lossAvg accAvg
    gradPAvg = num2cell(zeros(1,16));
    lossAvg = 0;
    accAvg = 0;
    for subiter=1:BIGBATCHSIZE
        % ff
        sml.ff();
        
        % loss
        loss = sml.O;
        lossAvg = lossAvg+loss/BIGBATCHSIZE;
        
        % lbl groundtruthLbl
        [~,lbl] = max(permute(smn.O, [5 6 1 2 3 4]));
        [~, groundtruthLbl] = max(permute(y.O, [5 6 1 2 3 4]));
        fprintf('%d -> %d\n', [groundtruthLbl; lbl]);
        
        % acc
        acc = sum(lbl==groundtruthLbl) / MINIBATCHSIZE;
        accAvg = accAvg+acc/BIGBATCHSIZE;
        
        % bp
        sml.bp();
        
        % debugHist
        if(mod(iter,DISPLAY_HIST_INTERVAL)==0)
            if(~(exist('fig1', 'var') && ishghandle(fig1)))
                fig1 = figure('Name','O','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
            end
            if(~(exist('fig2', 'var') && ishghandle(fig2)))
                fig2 = figure('Name','P','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
            end
            if(~(exist('fig3', 'var') && ishghandle(fig3)))
                fig3 = figure('Name','gradO','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
            end
            if(~(exist('fig4', 'var') && ishghandle(fig4)))
                fig4 = figure('Name','gradP','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
            end
            sml.debugHist(fig1, fig2, fig3, fig4, 8, 7, 0);
            drawnow;
        end
        
        gradPAvg{1} = gradPAvg{1}+c11.gradP/BIGBATCHSIZE;
        gradPAvg{2} = gradPAvg{2}+b11.gradP/BIGBATCHSIZE;
        gradPAvg{3} = gradPAvg{3}+c12.gradP/BIGBATCHSIZE;
        gradPAvg{4} = gradPAvg{4}+b12.gradP/BIGBATCHSIZE;
        gradPAvg{5} = gradPAvg{5}+c13.gradP/BIGBATCHSIZE;
        gradPAvg{6} = gradPAvg{6}+b13.gradP/BIGBATCHSIZE;
        gradPAvg{7} = gradPAvg{7}+c14.gradP/BIGBATCHSIZE;
        gradPAvg{8} = gradPAvg{8}+b14.gradP/BIGBATCHSIZE;
        gradPAvg{9} = gradPAvg{9}+c21.gradP/BIGBATCHSIZE;
        gradPAvg{10} = gradPAvg{10}+b21.gradP/BIGBATCHSIZE;
        gradPAvg{11} = gradPAvg{11}+c22.gradP/BIGBATCHSIZE;
        gradPAvg{12} = gradPAvg{12}+b22.gradP/BIGBATCHSIZE;
        gradPAvg{13} = gradPAvg{13}+f7.gradP/BIGBATCHSIZE;
        gradPAvg{14} = gradPAvg{14}+b7.gradP/BIGBATCHSIZE;
    end
    
    % display losses accs
    fprintf('lossAvg: %e, accAvg: %.3f %%\n', lossAvg, accAvg*100);
    losses{iter} = lossAvg;
    accs{iter} = accAvg;
    if(mod(iter, DISPLAY_LOSSES_ACCS_INTERVAL)==0)
        if(~(exist('fig0', 'var') && ishghandle(fig0)))
            fig0 = figure('Name','losses accs','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
        end
        set(0, 'CurrentFigure', fig0);
        subplot(1,2,1);
        semilogy(1:iter, cell2mat(losses), '+');
        title('losses');
        subplot(1,2,2);
        plot(1:iter, cell2mat(accs)*100, '+');
        ylim([0 100]);
        title('accs');
        drawnow;
    end
    
    % update
    v{1} = MOM*v{1} + LR*(-gradPAvg{1}-WD*c11.P);
    v{2} = MOM*v{2} + LR*(-gradPAvg{2});
    v{3} = MOM*v{3} + LR*(-gradPAvg{3}-WD*c12.P);
    v{4} = MOM*v{4} + LR*(-gradPAvg{4});
    v{5} = MOM*v{5} + LR*(-gradPAvg{5}-WD*c13.P);
    v{6} = MOM*v{6} + LR*(-gradPAvg{6});
    v{7} = MOM*v{7} + LR*(-gradPAvg{7}-WD*c14.P);
    v{8} = MOM*v{8} + LR*(-gradPAvg{8});
    v{9} = MOM*v{9} + LR*(-gradPAvg{9}-WD*c21.P);
    v{10} = MOM*v{10} + LR*(-gradPAvg{10});
    v{11} = MOM*v{11} + LR*(-gradPAvg{11}-WD*c22.P);
    v{12} = MOM*v{12} + LR*(-gradPAvg{12});
    v{13} = MOM*v{13} + LR*(-gradPAvg{13}-WD* f7.P);
    v{14} = MOM*v{14} + LR*(-gradPAvg{14});
    
    c11.P = c11.P + v{1};
    b11.P = b11.P + v{2};
    c12.P = c12.P + v{3};
    b12.P = b12.P + v{4};
    c13.P = c13.P + v{5};
    b13.P = b13.P + v{6};
    c14.P = c14.P + v{7};
    b14.P = b14.P + v{8};
    c21.P = c21.P + v{9};
    b21.P = b21.P + v{10};
    c22.P = c22.P + v{11};
    b22.P = b22.P + v{12};
    f7.P  =  f7.P + v{13};
    b7.P  =  b7.P + v{14};
    
    % snapshot
    if(mod(iter,SNAPSHOT_INTERVAL)==0)
        fn = ['snapshot/snapshot-' num2str(iter) '.mat'];
        fprintf('save to %s\n', fn);
        params = sml.getPs();
        
        % 检查losses与accs尺寸一致
        assert(numel(losses)==numel(accs))
        
        % 检查v与params尺寸一致
        assert(numel(params)==numel(v))
        for i=1:numel(params)
            %size(params{i})
            %size(v{i})
            assert(all(size(params{i})==size(v{i})));
        end
        
        save(fn, 'params', 'losses', 'accs', 'v');
    end
    
    toc(t);
end
