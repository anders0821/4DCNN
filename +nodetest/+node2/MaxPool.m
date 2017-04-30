classdef MaxPool < handle
    properties
        pI;
        O;
        gradO;
    end
    
    methods
        function [this] = MaxPool(pI)
            this.pI = pI;
            [H, W, C, N] = size(this.pI.O);
            assert(mod(H,2)==0);
            assert(mod(W,2)==0);
            this.O = zeros([H/2 W/2 C N]);
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            
            [H, W, C, N] = size(this.pI.O);
            submat = zeros([H/2 W/2 C N 4]);
            submat(:,:,:,:,1) = this.pI.O(1:2:end, 1:2:end, :, :);
            submat(:,:,:,:,2) = this.pI.O(1:2:end, 2:2:end, :, :);
            submat(:,:,:,:,3) = this.pI.O(2:2:end, 1:2:end, :, :);
            submat(:,:,:,:,4) = this.pI.O(2:2:end, 2:2:end, :, :);
            this.O = max(submat, [], 5);
            
            %toc
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            % Çó[H/2 W/2 C N 4] [H/2*W/2*C*N 4]ÏÂµÄ×ø±ê
            [H, W, C, N] = size(this.pI.O);
            submat = zeros([H/2 W/2 C N 4]);
            submat(:,:,:,:,1) = this.pI.O(1:2:end, 1:2:end, :, :);
            submat(:,:,:,:,2) = this.pI.O(1:2:end, 2:2:end, :, :);
            submat(:,:,:,:,3) = this.pI.O(2:2:end, 1:2:end, :, :);
            submat(:,:,:,:,4) = this.pI.O(2:2:end, 2:2:end, :, :);
            [~, idx] = max(submat, [], 5);
            idx = sub2ind([H/2*W/2*C*N 4], 1:H/2*W/2*C*N, idx(:)');
            
            % this.gradOÐ´Èë[H/2 W/2 C N 4]Ï¡Êè¾ØÕó
            submat = zeros([H/2*W/2*C*N 4]);
            submat(idx) = this.gradO;
            submat = reshape(submat, [H/2 W/2 C N 4]);
            
            % [H/2 W/2 C N 4]Ï¡Êè¾ØÕó×ª[H W C N]Ï¡Êè¾ØÕó
            this.pI.gradO(1:2:end, 1:2:end, :, :) = submat(:,:,:,:,1);
            this.pI.gradO(1:2:end, 2:2:end, :, :) = submat(:,:,:,:,2);
            this.pI.gradO(2:2:end, 1:2:end, :, :) = submat(:,:,:,:,3);
            this.pI.gradO(2:2:end, 2:2:end, :, :) = submat(:,:,:,:,4);
            
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
