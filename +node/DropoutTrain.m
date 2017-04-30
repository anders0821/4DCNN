classdef DropoutTrain < handle
    properties
        preservedRatio;
        
        pI;
        O;
        gradO;
        mask;
    end
    
    methods
        function [this] = DropoutTrain(pI, preservedRatio)
            this.preservedRatio = preservedRatio;
            
            this.pI = pI;
            this.O = zeros(size(this.pI.O));
            this.gradO = zeros(size(this.O));
            this.mask = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            
            % 计算通道mask向量
            [H, W, D, T, C, N] = size(this.pI.O);
            preservedC = round(C*this.preservedRatio);
            this.mask = zeros([C N]);
            for i=1:N
                this.mask(:,i) = [ones(preservedC,1)
                    zeros(C-preservedC,1)];
                this.mask(:,i) = this.mask(randperm(C),i);
            end
            %this.mask
            this.mask = repmat(permute(this.mask, [3 4 5 6 1 2]), [size(this.pI.O,1) size(this.pI.O,2) size(this.pI.O,3) size(this.pI.O,4) 1 1]);
            
            % % 计算mask(旧)
            % this.mask = randi(2, [size(this.pI.O,5) size(this.pI.O,6)])-1;% 逐通道dropout
            % this.mask = repmat(permute(this.mask, [3 4 5 6 1 2]), [size(this.pI.O,1) size(this.pI.O,2) size(this.pI.O,3) size(this.pI.O,4) 1 1]);
            
            % mask
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
