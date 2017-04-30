classdef SoftMaxNorm < handle
    properties
        pI;
        O;
        gradO;
    end
    
    methods
        function [this] = SoftMaxNorm(pI)
            this.pI = pI;
            [H, W, C, N] = size(this.pI.O);
            assert(H==1);
            assert(W==1);
            this.O = zeros(size(pI.O));
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            [H, W, C, N] = size(this.pI.O);
            num = exp(this.pI.O);
            den = sum(exp(this.pI.O), 3);
            den = repmat(den, [1 1 size(num,3) 1]);
            this.O = num ./ den;
            %toc;
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            % 计算激活的梯度
            N = size(this.O, 4);
            for i=1:N
                vec = permute(this.pI.O(:,:,:,i), [3 4 1 2]);% 单个输入样本
                tmp = exp(vec) / sum(exp(vec));
                J = diag(tmp) - tmp*tmp.';% 单个输入样本对单个输出样本的J
                this.pI.gradO(:,:,:,i) = J' * permute(this.gradO(:,:,:,i), [3 4 1 2]);% 单个输入样本的梯度
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
            
            fprintf('	gradO');
            fprintf('	%d', size(this.gradO));
            fprintf('	(%d)', numel(this.gradO));
            fprintf('\n');
            
            total = total + numel(this.O)+numel(this.gradO);
            totalP = totalP + 0;
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
            
            if(~isempty(fig3))
                set(0, 'CurrentFigure', fig3);
                subplot(m,n,p);
                hist(this.gradO(:),100);
                title(class(this));
            end
        end
        
        function [Ps] = getPs(this)
            [Ps] = this.pI.getPs();
        end
        
        function [] = setPs(this, Ps)
            this.pI.setPs(Ps);
        end
    end
end
