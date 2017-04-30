clc;
clear;
close all;
drawnow;
rng default;

% 定义网络 测试FF BP用
util.dispspliter('init')
MINIBATCHSIZE = 4;
x = node.Data('./data/4D_DATA_FOR_NODE_TEST.mat', 'TESTX', MINIBATCHSIZE);
y = node.Data('./data/4D_DATA_FOR_NODE_TEST.mat', 'TESTY', MINIBATCHSIZE);

% % 测试minibatch滚动
% x.ff();
% y.ff();
% x.debugMemory();
% 
% x.ff();
% y.ff();
% x.debugMemory();
% 
% % 可视化数据层
% for i=1:MINIBATCHSIZE
%     for j=1:9
%         subplot(3,3,j);
%         imshow(x.O(:,:,j,1,1,i));
%         onehot = y.O(1,1,1,1,:,i);
%         [~, lbl] = max(onehot);
%         title(lbl);
%     end
%     drawnow;
%     pause(3);
% end
% return

c0 = node.LocalConnectValid(x, [3 3 1 1], 2);        c0.P = randn(size(c0.P))/11;
c1 = node.LocalConnectValid(c0, [3 3 3 3], 3);       c1.P = randn(size(c1.P))/10;
cv = node.LocalConnectValid(c1, [1 1 1 1], 4);       c1.P = randn(size(c1.P))/10;

b2 = node.Bias(cv);                     b2.P = randn(size(b2.P))/3;
t3 = node.LRelu(b2);
s4 = node.MaxPool(t3, [1 1 0 1]);
d5 = node.DropoutTrain(s4, 1); %4->3
d6 = node.DropoutTest(d5, 0.7);  %4->3

flat = node.Flatten(d6);
d7 = node.DropoutTrain(flat, 1); %96->42
d8 = node.DropoutTest(d7, 0.44);    %96->42

f9 = node.FullCon(d8, 7);               f9.P = randn(size(f9.P))/40;
b10 = node.Bias(f9);                    b10.P = randn(size(b10.P))/3;
smn = node.SoftMaxNorm(b10);
sml = node.SoftMaxLoss(smn, y);
sml.debugMemory();

% ff
util.dispspliter('ff')
sml.ff();
sml.ff();
sml.ff();
sml.debugMemory();

% check ff
util.dispspliter('check ff')
nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, x.O, c0.O, c1.O, cv.O, t3.O, s4.O, d5.O, d6.O, flat.O, d7.O, d8.O, b10.O, smn.O, y.O, sml.O, 0, []);

% bp
util.dispspliter('bp')
sml.bp();
sml.debugMemory();

% show hist
% fig1 = figure('Name','O','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
% fig2 = figure('Name','P','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
% fig3 = figure('Name','gradO','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
% fig4 = figure('Name','gradP','NumberTitle','off','units','normalized','outerposition',[0 0 1 1]);
% sml.debugHist(fig1, fig2, fig3, fig4, 8, 7, 0);

% % check bp
% util.dispspliter('check smn sml')
% % 换另一种SoftMatWithLoss的算法来验证 http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
% checkVal = zeros(1,1,1,1,7,MINIBATCHSIZE);
% for i=1:MINIBATCHSIZE
%     vec = permute(b10.O(:,:,:,:,:,i), [5 6 1 2 3 4]);
%     checkVal(:,:,:,:,:,i) = permute( exp(vec)/sum(exp(vec)), [3 4 5 6 1 2]);
% end
% lbl = h5read('./data/4D_DATA_FOR_NODE_TEST.mat', '/TESTY', [1 1 1 1 1 1+MINIBATCHSIZE*2], [1 1 1 1 7 MINIBATCHSIZE]);
% lbl = permute(lbl, [5 6 1 2 3 4]);
% [~, lbl] = max(lbl);
% idx = sub2ind(size(checkVal), ones(1,MINIBATCHSIZE), ones(1,MINIBATCHSIZE), ones(1,MINIBATCHSIZE), ones(1,MINIBATCHSIZE), lbl, 1:MINIBATCHSIZE);
% checkVal(idx) = checkVal(idx)-1;
% checkVal = checkVal / MINIBATCHSIZE;
% z = checkVal-b10.gradO;
% norm(z(:))

% util.dispspliter('check b10.gradP')
% eps = 1e-6;
% checkVal = zeros(size(b10.P));
% util.dispstat('','init');
% for i=1:numel(checkVal)
%     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal))); 
%     PL = b10.P;
%     PL(i) = PL(i)-eps;
%     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, PL, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
%     PR = b10.P;
%     PR(i) = PR(i)+eps;
%     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, PR, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
%     checkVal(i) = (LR-LL) / 2 / eps;
% end
% b10.gradP
% checkVal
% z = checkVal-b10.gradP;
% norm(z(:))

% util.dispspliter('check f9.gradP')
% eps = 1e-6;
% checkVal = zeros(size(f9.P));
% util.dispstat('','init');
% for i=1:numel(checkVal)
%     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));  
%     PL = f9.P;
%     PL(i) = PL(i)-eps;
%     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, PL, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
%     PR = f9.P;
%     PR(i) = PR(i)+eps;
%     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, PR, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
%     checkVal(i) = (LR-LL) / 2 / eps;
% end
% f9.gradP(:,1:10)
% checkVal(:,1:10)
% f9.gradP(:,end-9:end)
% checkVal(:,end-9:end)
% z = checkVal-f9.gradP;
% norm(z(:))

% util.dispspliter('check b2.gradP')
% eps = 1e-6;
% checkVal = zeros(size(b2.P));
% util.dispstat('','init');
% for i=1:numel(checkVal)
%     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));  
%     PL = b2.P;
%     PL(i) = PL(i)-eps;
%     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, PL, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
%     PR = b2.P;
%     PR(i) = PR(i)+eps;
%     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, PR, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
%     checkVal(i) = (LR-LL) / 2 / eps;
% end
% b2.gradP
% checkVal
% z = checkVal-b2.gradP;
% norm(z(:))

% util.dispspliter('check cv.gradP')
% eps = 1e-6;
% checkVal = zeros(size(cv.P));
% util.dispstat('','init');
% for i=1:numel(checkVal)
%     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));  
%     PL = cv.P;
%     PL(i) = PL(i)-eps;
%     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, PL, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
%     PR = cv.P;
%     PR(i) = PR(i)+eps;
%     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, PR, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
%     checkVal(i) = (LR-LL) / 2 / eps;
% end
% cv.gradP(:,:,1,1,1,1,1,1,1,1)
% checkVal(:,:,1,1,1,1,1,1,1,1)
% cv.gradP(:,:,end,end,end,end,end,end,end,end)
% checkVal(:,:,end,end,end,end,end,end,end,end)
% z = checkVal-cv.gradP;
% norm(z(:))

% % % util.dispspliter('check c1.gradP')
% % % eps = 1e-6;
% % % checkVal = zeros(size(c1.P));
% % % util.dispstat('','init');
% % % for i=1:numel(checkVal)
% % %     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));  
% % %     PL = c1.P;
% % %     PL(i) = PL(i)-eps;
% % %     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, PL, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
% % %     PR = c1.P;
% % %     PR(i) = PR(i)+eps;
% % %     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, PR, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
% % %     checkVal(i) = (LR-LL) / 2 / eps;
% % % end
% % % c1.gradP(:,:,1,1,1,1,1,1,1,1)
% % % checkVal(:,:,1,1,1,1,1,1,1,1)
% % % c1.gradP(:,:,end,end,end,end,end,end,end,end)
% % % checkVal(:,:,end,end,end,end,end,end,end,end)
% % % z = checkVal-c1.gradP;
% % % norm(z(:))

% % % util.dispspliter('check c0.gradP')
% % % eps = 1e-6;
% % % checkVal = zeros(size(c0.P));
% % % util.dispstat('','init');
% % % for i=1:numel(checkVal)
% % %     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));  
% % %     PL = c0.P;
% % %     PL(i) = PL(i)-eps;
% % %     LL = nodetest2.checkff(MINIBATCHSIZE, PL, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
% % %     PR = c0.P;
% % %     PR(i) = PR(i)+eps;
% % %     LR = nodetest2.checkff(MINIBATCHSIZE, PR, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 0, []);
% % %     checkVal(i) = (LR-LL) / 2 / eps;
% % % end
% % % c0.gradP(:,:,1,1,1,1,1,1,1,1)
% % % checkVal(:,:,1,1,1,1,1,1,1,1)
% % % c0.gradP(:,:,end,end,end,end,end,end,end,end)
% % % checkVal(:,:,end,end,end,end,end,end,end,end)
% % % z = checkVal-c0.gradP;
% % % norm(z(:))

% util.dispspliter('check s4.gradO')
% eps = 1e-6;
% checkVal = zeros(size(s4.gradO));
% util.dispstat('','init');
% for i=1:numel(checkVal)
%     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));  
%     PL = zeros(size(s4.gradO));
%     PL(i) = PL(i)-eps;
%     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 1, PL);
%     PR = zeros(size(s4.gradO));
%     PR(i) = PR(i)+eps;
%     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 1, PR);
%     checkVal(i) = (LR-LL) / 2 / eps;
% end
% s4.gradO(:,1:10)
% checkVal(:,1:10)
% s4.gradO(:,end-9:end)
% checkVal(:,end-9:end)
% z = checkVal-s4.gradO;
% norm(z(:))

% util.dispspliter('check t3.gradO')
% eps = 1e-6;
% checkVal = zeros(size(t3.gradO));
% util.dispstat('','init');
% for i=1:numel(checkVal)
%     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));  
%     PL = zeros(size(t3.gradO));
%     PL(i) = PL(i)-eps;
%     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 2, PL);
%     PR = zeros(size(t3.gradO));
%     PR(i) = PR(i)+eps;
%     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 2, PR);
%     checkVal(i) = (LR-LL) / 2 / eps;
% end
% t3.gradO(:,1:10)
% checkVal(:,1:10)
% t3.gradO(:,end-9:end)
% checkVal(:,end-9:end)
% z = checkVal-t3.gradO;
% norm(z(:))

% util.dispspliter('check c1.gradO')
% eps = 1e-6;
% checkVal = zeros(size(c1.O));
% util.dispstat('','init');
% for i=1:numel(checkVal)
%     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));  
%     PL = zeros(size(c1.O));
%     PL(i) = PL(i)-eps;
%     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 3, PL);
%     PR = zeros(size(c1.O));
%     PR(i) = PR(i)+eps;
%     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 3, PR);
%     checkVal(i) = (LR-LL) / 2 / eps;
% end
% c1.gradO(:,:,1,1,1,1)
% checkVal(:,:,1,1,1,1)
% c1.gradO(:,:,end,end,end,end)
% checkVal(:,:,end,end,end,end)
% z = checkVal-c1.gradO;
% norm(z(:))

% % % util.dispspliter('check c0.gradO')
% % % eps = 1e-6;
% % % checkVal = zeros(size(c0.gradO));
% % % util.dispstat('','init');
% % % for i=1:numel(checkVal)
% % %     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal)));
% % %     PL = zeros(size(c0.gradO));
% % %     PL(i) = PL(i)-eps;
% % %     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 4, PL);
% % %     PR = zeros(size(c0.gradO));
% % %     PR(i) = PR(i)+eps;
% % %     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 4, PR);
% % %     checkVal(i) = (LR-LL) / 2 / eps;
% % % end
% % % c0.gradO(:,:,1,1,1,1)
% % % checkVal(:,:,1,1,1,1)
% % % c0.gradO(:,:,end,end,end,end)
% % % checkVal(:,:,end,end,end,end)
% % % z = checkVal-c0.gradO;
% % % norm(z(:))

% % % util.dispspliter('check x.gradO')
% % % eps = 1e-6;
% % % checkVal = zeros(size(x.gradO));
% % % util.dispstat('','init');
% % % for i=1:numel(checkVal)
% % %     util.dispstat(sprintf('Processing %d%% %d/%d', round(i/(numel(checkVal))*100), i, numel(checkVal))); 
% % %     PL = zeros(size(x.gradO));
% % %     PL(i) = PL(i)-eps;
% % %     LL = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 5, PL);
% % %     PR = zeros(size(x.gradO));
% % %     PR(i) = PR(i)+eps;
% % %     LR = nodetest2.checkff(MINIBATCHSIZE, c0.P, c1.P, cv.P, b2.P, d5.mask, d7.mask, f9.P, b10.P, [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], 5, PR);
% % %     checkVal(i) = (LR-LL) / 2 / eps;
% % % end
% % % x.gradO(:,:,1,1,1,1)
% % % checkVal(:,:,1,1,1,1)
% % % x.gradO(:,:,end,end,end,end)
% % % checkVal(:,:,end,end,end,end)
% % % z = checkVal-x.gradO;
% % % norm(z(:))
