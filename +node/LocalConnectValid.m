% TODO: FF BP速度太慢

classdef LocalConnectValid < handle
    properties
        pI;
        P;
        O;
        gradP;
        gradO;
    end
    
    methods
        function [this] = LocalConnectValid(pI, KERNELWIDTHS, OUTPUTCHANNEL)
            assert(all(size(KERNELWIDTHS)==[1 4]));
            assert(all(mod(KERNELWIDTHS, 2)==1));
            assert(KERNELWIDTHS(1)<=size(pI.O,1));
            assert(KERNELWIDTHS(2)<=size(pI.O,2));
            assert(KERNELWIDTHS(3)<=size(pI.O,3));
            assert(KERNELWIDTHS(4)<=size(pI.O,4));
            % 初始化成员变量
            this.pI = pI;
            [H, W, D, T, INPUTCHANNEL, N] = size(this.pI.O);
            this.P = randn([KERNELWIDTHS INPUTCHANNEL OUTPUTCHANNEL H-KERNELWIDTHS(1)+1 W-KERNELWIDTHS(2)+1 D-KERNELWIDTHS(3)+1 T-KERNELWIDTHS(4)+1]) * 0.01;
            this.O = zeros([H-KERNELWIDTHS(1)+1 W-KERNELWIDTHS(2)+1 D-KERNELWIDTHS(3)+1 T-KERNELWIDTHS(4)+1 OUTPUTCHANNEL N]);
            this.gradP = zeros(size(this.P));
            this.gradO = zeros(size(this.O));
        end
        
        function [B] = conv4ValidNoShare(this, A, K)
            [KERNELWIDTH1, KERNELWIDTH2, KERNELWIDTH3, KERNELWIDTH4, INPUTCHANNEL, OUTPUTCHANNEL, H, W, D, T] = size(K);
            % assert(INPUTCHANNEL==1);
            % assert(OUTPUTCHANNEL==1);
            B = zeros([H W D T]);
            
            % size(A)
            % size(K)
            % size(B)
            
            for t=1:T
            for d=1:D
            for w=1:W
            for h=1:H
                subA = A(h:h+KERNELWIDTH1-1, w:w+KERNELWIDTH2-1, d:d+KERNELWIDTH3-1, t:t+KERNELWIDTH4-1);
                subK = K(:,:,:,:,1,1,h,w,d,t);
                B(h,w,d,t) = dot(subA(:), subK(:));
            end
            end
            end
            end
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            [H, W, D, T, INPUTCHANNEL, N] = size(this.pI.O);
            OUTPUTCHANNEL = size(this.P, 6);
            
            this.O = zeros([H-size(this.P,1)+1 W-size(this.P,2)+1 D-size(this.P,3)+1 T-size(this.P,4)+1 OUTPUTCHANNEL N]);
            for i=1:N
                for j=1:OUTPUTCHANNEL
                    for k=1:INPUTCHANNEL
                        tmp = this.conv4ValidNoShare(this.pI.O(:,:,:,:,k,i), this.P(:,:,:,:,k,j,:,:,:,:));
                        this.O(:,:,:,:,j,i) = this.O(:,:,:,:,j,i) + tmp;
                    end
                end
            end
            
            %toc;
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            [H, W, D, T, INPUTCHANNEL, N] = size(this.pI.O);
            [KERNELWIDTH1, KERNELWIDTH2, KERNELWIDTH3, KERNELWIDTH4, ~, OUTPUTCHANNEL, H2, W2, D2, T2] = size(this.P);
            KERNELWIDTHS = [KERNELWIDTH1, KERNELWIDTH2, KERNELWIDTH3, KERNELWIDTH4];
            
            % 计算激活的梯度
            % 扩展P
            assert(all(mod(KERNELWIDTHS, 2)==1));
            extBrd = (KERNELWIDTHS-1) / 2;
            rotExtP = zeros([KERNELWIDTH1, KERNELWIDTH2, KERNELWIDTH3, KERNELWIDTH4, INPUTCHANNEL, OUTPUTCHANNEL H2+extBrd(1)*2 W2+extBrd(2)*2 D2+extBrd(3)*2 T2+extBrd(4)*2]);
            % size(rotExtP)
            % size(this.P)
            rotExtP(:,:,:,:,:,:,1+extBrd(1):end-extBrd(1), 1+extBrd(2):end-extBrd(2), 1+extBrd(3):end-extBrd(3), 1+extBrd(4):end-extBrd(4)) = this.P;
            rotExtP = flip(rotExtP, 1);
            rotExtP = flip(rotExtP, 2);
            rotExtP = flip(rotExtP, 3);
            rotExtP = flip(rotExtP, 4);
            % 扩展gradO
            extGradO = zeros([H2+extBrd(1)*4 W2+extBrd(2)*4 D2+extBrd(3)*4 T2+extBrd(4)*4, OUTPUTCHANNEL, N]);
            % size(extGradO)
            % size(this.gradO)
            extGradO(1+extBrd(1)*2:end-extBrd(1)*2, 1+extBrd(2)*2:end-extBrd(2)*2, 1+extBrd(3)*2:end-extBrd(3)*2, 1+extBrd(4)*2:end-extBrd(4)*2, :, :) = this.gradO;
            % 反向卷积
            this.pI.gradO = zeros(size(this.pI.O));
            for i=1:N
                for j=1:INPUTCHANNEL
                    for k=1:OUTPUTCHANNEL
                        tmp = this.conv4ValidNoShare(extGradO(:,:,:,:,k,i), rotExtP(:,:,:,:,j,k,:,:,:,:));
                        this.pI.gradO(:,:,:,:,j,i) = this.pI.gradO(:,:,:,:,j,i) + tmp;
                    end
                end
            end
            
            % 计算参数的梯度
            this.gradP = zeros(size(this.P));
            % size(this.gradO)
            % size(this.pI.O)
            % size(this.gradP)
            for i=1:N
                for j=1:OUTPUTCHANNEL
                    for k=1:INPUTCHANNEL
                        for w4=1:KERNELWIDTH4
                        for w3=1:KERNELWIDTH3
                        for w2=1:KERNELWIDTH2
                        for w1=1:KERNELWIDTH1
                            tmp = this.gradO(:,:,:,:,j,i) .* this.pI.O(1+w1-1:H2+w1-1, 1+w2-1:W2+w2-1, 1+w3-1:D2+w3-1, 1+w4-1:T2+w4-1, k,i);
                            this.gradP(w1,w2,w3,w4,k,j,:,:,:,:) = this.gradP(w1,w2,w3,w4,k,j,:,:,:,:) + permute(tmp, [5 6 7 8 9 10 1 2 3 4]);
                        end
                        end
                        end
                        end
                    end
                end
            end
            
            %toc;
            
            this.pI.bp();
        end
        
        function [total, totalP] = debugMemory(this)
            [total, totalP] = this.pI.debugMemory();
            
            disp(util.funname());
            
            fprintf('	O');
            fprintf('	%d', size(this.O));
            fprintf('	(%d)', numel(this.O));
            fprintf('\n');
            
            fprintf('	P');
            fprintf('	%d', size(this.P));
            fprintf('	(%d)', numel(this.P));
            fprintf('\n');
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            fprintf('	gradP');
            fprintf('	%d', size(this.gradP));
            fprintf('	(%d)', numel(this.gradP));
            fprintf('\n');
            
            total = total + numel(this.O)+numel(this.gradO) + numel(this.P)+numel(this.gradP);
            totalP = totalP + numel(this.gradP);
        end
        
        function [p] = debugHist(this, fig1, fig2, fig3, fig4, m, n, p)
            p = this.pI.debugHist(fig1, fig2, fig3, fig4, m, n, p);
            
            p = p+1;
            
            if(~isempty(fig1))
                set(0, 'CurrentFigure', fig1);
                subplot(m,n,p);
                hist(this.O(:),100);
                title(class(this));
            end
            
            if(~isempty(fig2))
                set(0, 'CurrentFigure', fig2);
                subplot(m,n,p);
                hist(this.P(:),100);
                title(class(this));
            end
            
            if(~isempty(fig3))
                set(0, 'CurrentFigure', fig3);
                subplot(m,n,p);
                hist(this.gradO(:),100);
                title(class(this));
            end
            
            if(~isempty(fig4))
                set(0, 'CurrentFigure', fig4);
                subplot(m,n,p);
                hist(this.gradP(:),100);
                title(class(this));
            end
        end
        
        function [Ps] = getPs(this)
            [Ps] = this.pI.getPs();
            Ps{end+1} = this.P;
        end
        
        function [] = setPs(this, Ps)
            this.P = Ps{end};
            Ps = Ps(1:end-1);
            
            this.pI.setPs(Ps);
        end
    end
end
