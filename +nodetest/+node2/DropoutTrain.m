classdef DropoutTrain < handle
    properties
        pI;
        O;
        gradO;
        mask;
    end
    
    methods
        function [this] = DropoutTrain(pI)
            this.pI = pI;
            this.O = zeros(size(this.pI.O));
            this.gradO = zeros(size(this.O));
            this.mask = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            % this.mask = randi(2, size(this.pI.O))-1;% 逐位置/通道dropout
            this.mask = randi(2, [size(this.pI.O,3) size(this.pI.O,4)])-1;% 逐通道dropout
            this.mask = repmat(permute(this.mask, [3 4 1 2]), [size(this.pI.O,1) size(this.pI.O,2) 1 1]);
            this.O = this.pI.O .* this.mask;
            %toc
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            this.pI.gradO = this.gradO .* this.mask;
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
            
            fprintf('	mask');
            fprintf('	%d', size(this.mask));
            fprintf('	(%d)', numel(this.mask));
            fprintf('\n');
            
            total = total + numel(this.O)+numel(this.gradO)+numel(this.mask);
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
