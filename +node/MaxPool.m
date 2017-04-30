classdef MaxPool < handle
    properties
        isSub;
        
        pI;
        O;
        gradO;
    end
    
    methods
        function [this] = MaxPool(pI, isSub)
            this.pI = pI;
            assert(all(size(isSub)==[1 4]));
            this.isSub = isSub;
            
            % HWDTCN -> H2W2D2T2CN
            [H, W, D, T, C, N] = size(this.pI.O);
            if(isSub(1))
                assert(mod(H,2)==0);
                H2 = H/2;
            else
                H2 = H;
            end
            if(isSub(2))
                assert(mod(W,2)==0);
                W2 = W/2;
            else
                W2 = W;
            end
            if(isSub(3))
            	assert(mod(D,2)==0);
                D2 = D/2;
            else
                D2 = D;
            end
            if(isSub(4))
                assert(mod(T,2)==0);
                T2  = T/2;
            else
                T2  = T;
            end
            this.O = zeros([H2 W2 D2 T2 C N]);
            this.gradO = zeros(size(this.O));
        end
        
        function [] = ff(this)
            this.pI.ff();
            
            %fprintf('%s\t', util.funname());
            %tic;
            
            % 子矩阵元素值相同情况下
            % 取最大优先顺序为 [1 3
            %                  2 4]
            % HWDTCN -> H2W2D2T2CN
            [H, W, D, T, C, N] = size(this.pI.O);
            this.O = this.pI.O;
            if(this.isSub(1))
                part1 = this.O(1:2:end, :, :, :, :, :);
                part2 = this.O(2:2:end, :, :, :, :, :);
                this.O = max(part1, part2);
            end
            if(this.isSub(2))
                part1 = this.O(:, 1:2:end, :, :, :, :);
                part2 = this.O(:, 2:2:end, :, :, :, :);
                this.O = max(part1, part2);
            end
            if(this.isSub(3))
                part1 = this.O(:, :, 1:2:end, :, :, :);
                part2 = this.O(:, :, 2:2:end, :, :, :);
                this.O = max(part1, part2);
            end
            if(this.isSub(4))
                part1 = this.O(:, :, :, 1:2:end, :, :);
                part2 = this.O(:, :, :, 2:2:end, :, :);
                this.O = max(part1, part2);
            end
            
            %toc
        end
        
        function [] = bp(this)
            %fprintf('%s\t', util.funname());
            %tic;
            
            % 正向求switch
            % HWDTCN -> H2W2D2T2CN
            [H, W, D, T, C, N] = size(this.pI.O);
            tmp = this.pI.O;
            if(this.isSub(1))
                part1 = tmp(1:2:end, :, :, :, :, :);
                part2 = tmp(2:2:end, :, :, :, :, :);
                tmp = max(part1, part2);
                switch1 = part1>=part2;
            end
            if(this.isSub(2))
                part1 = tmp(:, 1:2:end, :, :, :, :);
                part2 = tmp(:, 2:2:end, :, :, :, :);
                tmp = max(part1, part2);
                switch2 = part1>=part2;
            end
            if(this.isSub(3))
                part1 = tmp(:, :, 1:2:end, :, :, :);
                part2 = tmp(:, :, 2:2:end, :, :, :);
                tmp = max(part1, part2);
                switch3 = part1>=part2;
            end
            if(this.isSub(4))
                part1 = tmp(:, :, :, 1:2:end, :, :);
                part2 = tmp(:, :, :, 2:2:end, :, :);
                tmp = max(part1, part2);
                switch4 = part1>=part2;
            end
            
            % 逆向求梯度
            % H2W2D2T2CN -> HWDTCN
            this.pI.gradO = this.gradO;
            if(this.isSub(4))
                [H, W, D, T, C, N] = size(this.pI.gradO);
                part1 = this.pI.gradO.*switch4;
                part2 = this.pI.gradO.*(1-switch4);
                part1 = permute(part1, [1 2 3 7 4 5 6]);
                part2 = permute(part2, [1 2 3 7 4 5 6]);
                this.pI.gradO = cat(4, part1, part2);
                this.pI.gradO = reshape(this.pI.gradO, [H W D T*2 C N]);
            end
            if(this.isSub(3))
                [H, W, D, T, C, N] = size(this.pI.gradO);
                part1 = this.pI.gradO.*switch3;
                part2 = this.pI.gradO.*(1-switch3);
                part1 = permute(part1, [1 2 7 3 4 5 6]);
                part2 = permute(part2, [1 2 7 3 4 5 6]);
                this.pI.gradO = cat(3, part1, part2);
                this.pI.gradO = reshape(this.pI.gradO, [H W D*2 T C N]);
            end
            if(this.isSub(2))
                [H, W, D, T, C, N] = size(this.pI.gradO);
                part1 = this.pI.gradO.*switch2;
                part2 = this.pI.gradO.*(1-switch2);
                part1 = permute(part1, [1 7 2 3 4 5 6]);
                part2 = permute(part2, [1 7 2 3 4 5 6]);
                this.pI.gradO = cat(2, part1, part2);
                this.pI.gradO = reshape(this.pI.gradO, [H W*2 D T C N]);
            end
            if(this.isSub(1))
                [H, W, D, T, C, N] = size(this.pI.gradO);
                part1 = this.pI.gradO.*switch1;
                part2 = this.pI.gradO.*(1-switch1);
                part1 = permute(part1, [7 1 2 3 4 5 6]);
                part2 = permute(part2, [7 1 2 3 4 5 6]);
                this.pI.gradO = cat(1, part1, part2);
                this.pI.gradO = reshape(this.pI.gradO, [H*2 W D T C N]);
            end
            
            %toc;
            
            this.pI.bp();
        end
        
        function [total, totalP] = debugMemory(this)
            [total, totalP] = this.pI.debugMemory();
            
            disp(util.funname());
            
            fprintf('	isSub');
            fprintf('	%d', this.isSub);
            fprintf('\n');
            
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
