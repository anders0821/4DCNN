classdef Conv < handle
    properties
        pI;
        P;
        O;
        gradP;
        gradO;
    end
    
    methods
        function [this] = Conv(pI, KERNELWIDTH, OUTPUTCHANNEL)
            assert(mod(KERNELWIDTH, 2)==1);
            % 初始化成员变量
            this.pI = pI;
            [H, W, INPUTCHANNEL, N] = size(this.pI.O);
            this.P = randn([KERNELWIDTH KERNELWIDTH INPUTCHANNEL OUTPUTCHANNEL]) * 0.01;
            this.O = zeros([H W OUTPUTCHANNEL N]);
            this.gradP = zeros(size(this.P));
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            [H, W, INPUTCHANNEL, N] = size(this.pI.O);
            OUTPUTCHANNEL = size(this.P, 4);
            
            this.O = zeros([H W OUTPUTCHANNEL N]);
            for i=1:N
                for j=1:OUTPUTCHANNEL
                    for k=1:INPUTCHANNEL
                        tmp = conv2(this.pI.O(:,:,k,i), this.P(:,:,k,j), 'same');
                        this.O(:,:,j,i) = this.O(:,:,j,i) + tmp;
                    end
                end
            end
            
            %toc;
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            [H, W, INPUTCHANNEL, N] = size(this.pI.O);
            [KERNELWIDTH, ~, ~, OUTPUTCHANNEL] = size(this.P);
            
            % 计算激活的梯度
            rotP = this.P;
            for i=1:OUTPUTCHANNEL
                for j=1:INPUTCHANNEL
                    rotP(:,:,j,i) = rot90(rotP(:,:,j,i), 2);
                end
            end
            this.pI.gradO = zeros(size(this.pI.O));
            for i=1:N
                for j=1:INPUTCHANNEL
                    for k=1:OUTPUTCHANNEL
                        tmp = conv2(this.gradO(:,:,k,i), rotP(:,:,j,k), 'same');
                        this.pI.gradO(:,:,j,i) = this.pI.gradO(:,:,j,i) + tmp;
                    end
                end
            end
            
            % 计算参数的梯度
            rotI = this.pI.O;
            for i=1:N
                for j=1:INPUTCHANNEL
                    rotI(:,:,j,i) = rot90(rotI(:,:,j,i), 2);
                end
            end
            assert(mod(KERNELWIDTH, 2)==1);
            extBrd = (KERNELWIDTH-1) / 2;
            extGradO = zeros([H+extBrd*2 W+extBrd*2 OUTPUTCHANNEL N]);
            extGradO(1+extBrd:end-extBrd, 1+extBrd:end-extBrd, :, :) = this.gradO;
            this.gradP = zeros(size(this.P));
            for i=1:N
                for j=1:OUTPUTCHANNEL
                    for k=1:INPUTCHANNEL
                        tmp = conv2(extGradO(:,:,j,i), rotI(:,:,k,i), 'valid');
                        this.gradP(:,:,k,j) = this.gradP(:,:,k,j) + tmp;
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
