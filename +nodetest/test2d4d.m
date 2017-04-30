clc;
clear;
close all;
drawnow;
rng default;

% ∂®“ÂÕ¯¬Á ≤‚ ‘FF BP”√
util.dispspliter('init')
MINIBATCHSIZE = 4;


% 4D
x = node.Data('./data/2D_DATA_FOR_NODE_TEST.mat', 'TESTX', MINIBATCHSIZE);
y = node.Data('./data/2D_DATA_FOR_NODE_TEST.mat', 'TESTY', MINIBATCHSIZE);
size(x.O)
size(y.O)
c0 = node.ConvParallelStopBp(x, [11 11 1 1], 2);        c0.P = randn(size(c0.P))/11;
c1 = node.ConvParallel(c0, [3 3 1 1], 3);       c1.P = randn(size(c1.P))/10;
b2 = node.Bias(c1);                     b2.P = zeros(size(b2.P))/3;
t3 = node.Tanh(b2);
s4 = node.MaxPool(t3, [1 1 0 0]);
d5 = node.DropoutTrain(s4);
d6 = node.DropoutTest(d5);
flat = node.Flatten(d6);
d7 = node.DropoutTrain(flat);
d8 = node.DropoutTest(d7);
f9 = node.FullCon(d8, 7);               f9.P = randn(size(f9.P))/40;
b10 = node.Bias(f9);                    b10.P = zeros(size(b10.P))/3;
smn = node.SoftMaxNorm(b10);
sml = node.SoftMaxLoss(smn, y);

%
sml.ff();
sml.bp();
sml.ff();
sml.bp();
sml.ff();
sml.bp();

% 2D
rng default;
xA = nodetest.node2.Data(permute(x.O, [1 2 5 6 3 4]), MINIBATCHSIZE);
yA = nodetest.node2.Data(permute(y.O, [1 2 5 6 3 4]), MINIBATCHSIZE);
size(xA.O)
size(yA.O)
c0A = nodetest.node2.ConvParallelStopBp(xA, 11, 2);        c0A.P = randn(size(c0A.P))/11;
c1A = nodetest.node2.ConvParallel(c0A, 3, 3);       c1A.P = randn(size(c1A.P))/10;
b2A = nodetest.node2.Bias(c1A);                     b2A.P = zeros(size(b2A.P))/3;
t3A = nodetest.node2.Tanh(b2A);
s4A = nodetest.node2.MaxPool(t3A);
d5A = nodetest.node2.DropoutTrain(s4A);
d6A = nodetest.node2.DropoutTest(d5A);
flatA = nodetest.node2.Flatten(d6A);
d7A = nodetest.node2.DropoutTrain(flatA);
d8A = nodetest.node2.DropoutTest(d7A);
f9A = nodetest.node2.FullCon(d8A, 7);               f9A.P = randn(size(f9A.P))/40;
b10A = nodetest.node2.Bias(f9A);                    b10A.P = zeros(size(b10A.P))/3;
smnA = nodetest.node2.SoftMaxNorm(b10A);
smlA = nodetest.node2.SoftMaxLoss(smnA, yA);

%
smlA.ff();
smlA.bp();
smlA.ff();
smlA.bp();
smlA.ff();
smlA.bp();

%
util.dispspliter('check O')
all(x.O(:)==xA.O(:))
all(y.O(:)==yA.O(:))
all(c0.O(:)==c0A.O(:))
all(c1.O(:)==c1A.O(:))
all(b2.O(:)==b2A.O(:))
all(t3.O(:)==t3A.O(:))
all(s4.O(:)==s4A.O(:))
all(d5.O(:)==d5A.O(:))
all(d6.O(:)==d6A.O(:))
all(flat.O(:)==flatA.O(:))
all(d7.O(:)==d7A.O(:))
all(d8.O(:)==d8A.O(:))
all(f9.O(:)==f9A.O(:))
all(b10.O(:)==b10A.O(:))
all(smn.O(:)==smnA.O(:))
all(sml.O(:)==smlA.O(:))

%
util.dispspliter('check gradO')
all(x.gradO(:)==xA.gradO(:))
all(y.gradO(:)==yA.gradO(:))
all(c0.gradO(:)==c0A.gradO(:))
all(c1.gradO(:)==c1A.gradO(:))
all(b2.gradO(:)==b2A.gradO(:))
all(t3.gradO(:)==t3A.gradO(:))
all(s4.gradO(:)==s4A.gradO(:))
all(d5.gradO(:)==d5A.gradO(:))
all(d6.gradO(:)==d6A.gradO(:))
all(flat.gradO(:)==flatA.gradO(:))
all(d7.gradO(:)==d7A.gradO(:))
all(d8.gradO(:)==d8A.gradO(:))
all(f9.gradO(:)==f9A.gradO(:))
all(b10.gradO(:)==b10A.gradO(:))
all(smn.gradO(:)==smnA.gradO(:))

%
util.dispspliter('check P')
all(c0.P(:)==c0A.P(:))
all(c1.P(:)==c1A.P(:))
all(b2.P(:)==b2A.P(:))
all(f9.P(:)==f9A.P(:))
all(b10.P(:)==b10A.P(:))

%
util.dispspliter('check gradP')
all(c0.gradP(:)==c0A.gradP(:))
all(c1.gradP(:)==c1A.gradP(:))
all(b2.gradP(:)==b2A.gradP(:))
all(f9.gradP(:)==f9A.gradP(:))
all(b10.gradP(:)==b10A.gradP(:))
