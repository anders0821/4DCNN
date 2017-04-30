classdef SoftMaxLoss < handle
    properties
        pI;
        pY;
        O;
    end
    
    methods
        function [this] = SoftMaxLoss(pI, pY)
            this.pI = pI;
            [H, W, C, N] = size(this.pI.O);
            assert(H==1);
            assert(W==1);
            this.pY = pY;
            [H, W, C, N] = size(this.pY.O);
            assert(H==1);
            assert(W==1);
            assert(all( size(this.pI.O)==size(this.pY.O) ));
            
            this.O = zeros(1);
        end
        
        function [] = ff(this)
            this.pI.ff();
            this.pY.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            this.O = -log(this.pI.O) .* this.pY.O;
            this.O = sum(this.O(:)) / size(this.O, 4);
            %toc;
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            % 计算激活的梯度
            % 只有LOSS层的激活的梯度需要在N上平均
            % 在其他层中，由于FF时候分N组独立计算，激活的梯度亦分N组独立计算，参数的梯度需要在N上求和
            N = size(this.pI.O, 4);
            this.pI.gradO = this.pY.O ./ this.pI.O / (-N);
            
            %toc;
            
            this.pI.bp();
        end
        
        function [total, totalP] = debugMemory(this)
            [total, totalP] = this.pI.debugMemory();
            
            [total2, totalP2] = this.pY.debugMemory();
            total = total + total2;
            totalP = totalP + totalP2;
            
            disp(util.funname());
            
            fprintf('	O');
            fprintf('	%d', size(this.O));
            fprintf('			(%d)', numel(this.O));
            fprintf('\n');
            
            total = total + numel(this.O);
            totalP = totalP + 0;
        end
        
        function [p] = debugHist(this, fig1, fig2, fig3, fig4, m, n, p)
            p = this.pI.debugHist(fig1, fig2, fig3, fig4, m, n, p);
            
            p = this.pY.debugHist(fig1, fig2, fig3, fig4, m, n, p);
            
            p = p+1;
            
            if(~isempty(fig1))
                set(0, 'CurrentFigure', fig1);
                subplot(m,n,p);
                hist(this.O(:),100);
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
