classdef DropoutTest < handle
    properties
        preservedRatio;
        
        pI;
        O;
        gradO;
    end
    
    methods
        function [this] = DropoutTest(pI, preservedRatio)
            this.preservedRatio = preservedRatio;
            
            this.pI = pI;
            this.O = zeros(size(this.pI.O));
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            [H, W, D, T, C, N] = size(this.pI.O);
            preservedC = round(C*this.preservedRatio);
            extractPreservedRatioRatio = preservedC/C;
            
            %fprintf('%s\t', util.funname());
            %tic;
            this.O = this.pI.O .* extractPreservedRatioRatio;
            %toc
        end
        
        function [] = bp(this)
            [H, W, D, T, C, N] = size(this.pI.O);
            preservedC = round(C*this.preservedRatio);
            extractPreservedRatioRatio = preservedC/C;
            
            %fprintf('%s\t', util.funname());
            %tic;
            this.pI.gradO = this.gradO .* extractPreservedRatioRatio;
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
