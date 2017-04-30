classdef FullCon < handle
    properties
        pI;
        P;
        O;
        gradP;
        gradO;
    end
    
    methods
        function [this] = FullCon(pI, OUTPUTCHANNEL)
            % 初始化成员变量
            this.pI = pI;
            [H, W, INPUTCHANNEL, N] = size(this.pI.O);
            assert(H==1);
            assert(W==1);
            
            this.P = randn([OUTPUTCHANNEL INPUTCHANNEL]) * 0.01;
            this.O = zeros([1 1 OUTPUTCHANNEL N]);
            this.gradP = zeros(size(this.P));
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            this.O = this.P * permute(this.pI.O, [3 4 1 2]);
            this.O = permute(this.O, [3 4 1 2]);
            %toc;
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            % 计算激活的梯度
            % N = size(this.O, 4);
            % for i=1:N
            %     J = this.P;
            %     this.pI.gradO(:,:,:,i) = J' * permute(this.gradO(:,:,:,i), [3 4 1 2]);% 单个输入样本的梯度
            % end
            
            % 快速计算激活的梯度
            J = this.P;
            this.pI.gradO = J' * permute(this.gradO, [3 4 1 2]);% 单个输入样本的梯度
            this.pI.gradO = permute(this.pI.gradO, [3 4 1 2]);
            
            % 计算参数的梯度
            [H, W, INPUTCHANNEL, N] = size(this.pI.O);
            [H, W, OUTPUTCHANNEL, N] = size(this.O);
            this.gradP = zeros(OUTPUTCHANNEL, INPUTCHANNEL);
            for i=1:N
                this.gradP = this.gradP + permute(this.gradO(:,:,:,i), [3 4 1 2]) * permute(this.pI.O(:,:,:,i), [3 4 1 2])';
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
            fprintf('			(%d)', numel(this.P));
            fprintf('\n');
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            fprintf('	gradP');
            fprintf('	%d', size(this.gradP));
            fprintf('			(%d)', numel(this.gradP));
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
