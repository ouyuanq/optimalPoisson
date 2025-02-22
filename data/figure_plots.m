% plotting spectrum of second-order differential operator approximated by
% ultraspherical spectral method and its estimations derived from
% characteristic polynomials

syms n
% coefficients of characteristic polynomial
ak = @(k) 2^(2*k)*factorial(2*k+1)*gamma(3+n+2*k)/(gamma(4+4*k)*factorial(n-2*k-1));
bk = @(k) 2^(2*k)*factorial(2*k+1)*gamma(4+n+2*k)/(gamma(4+4*k)*factorial(n-2*k));

a0 = ak(0); b0 = bk(0);
a1 = ak(1); b1 = bk(1);
a2 = ak(2); b2 = bk(2);
a3 = ak(3); b3 = bk(3);

an2m1 = ak(n/2-1); bn2 = bk(n/2);
an2m2 = ak(n/2-2); bn2m1 = bk(n/2-1);
an2m3 = ak(n/2-3); bn2m2 = bk(n/2-2);
an2m4 = ak(n/2-4); bn2m3 = bk(n/2-3);

% root bounds of Newton
% square sum
upperb2 = simplify(expand( (b1/b0)^2 - 2*(b2/b0) ));
uppera2 = simplify(expand( (a1/a0)^2 - 2*(a2/a0) ));
lowerb2 = simplify(expand( (bn2m1/bn2)^2 - 2*(bn2m2/bn2) ));
lowera2 = simplify(expand( (an2m2/an2m1)^2 - 2*(an2m3/an2m1) ));
% % cubic sum
% upperb3 = simplify(expand( (b1/b0)^3 - 3*(b1/b0)*(b2/b0)+3*(b3/b0) ));
% uppera3 = simplify(expand( (a1/a0)^3 - 3*(a1/a0)*(a2/a0)+3*(a3/a0) ));
% lowerb3 = simplify(expand( (bn2m1/bn2)^3 - 3*(bn2m1/bn2)*(bn2m2/bn2)+3*(bn2m3/bn2) ));
% lowera3 = simplify(expand( (an2m2/an2m1)^3 - 3*(an2m2/an2m1)*(an2m3/an2m1)...
%     +3*(an2m4/an2m1) ));

% numerical results
nvec = 2 .^ (3:11);
mineig = zeros(length(nvec), 1);
maxeig = zeros(length(nvec), 1);
upperbound2 = zeros(length(nvec), 1);
upperbound3 = zeros(length(nvec), 1);
lowerbound2 = zeros(length(nvec), 1);
lowerbound3 = zeros(length(nvec), 1);
eigcond = zeros(length(nvec), 1);
for i = 1:length(nvec)
    nnow = nvec(i);
    D = ultraS.diffmat(nnow+2, 2);  % differential operator
    S = ultraS.convertmat(nnow+2, 0, 1);  % conversion operator
    R = spdiags([-ones(nnow+2, 1) ones(nnow+2, 1)], [-2; 0], nnow+2, nnow);  % transformation operator
    
    A = D*R; B = S*R;
    K = full(B(1:nnow, 1:nnow)) \ full(A(1:nnow, 1:nnow));
    
    [VK, DK] = eig(K);
    eK = sort(eig(-K));
    mineig(i) = min(eK);
    maxeig(i) = max(eK);
    
    upperbound2(i) = max(abs(subs(upperb2, n, nvec(i)+1))^(1/2), ...
        abs(subs(uppera2, n, nvec(i)+1))^(1/2));
    % upperbound3(i) = max(abs(subs(upperb3, n, nvec(i)+1))^(1/3), ...
    %     abs(subs(uppera3, n, nvec(i)+1))^(1/3));
    lowerbound2(i) = min(abs(subs(lowerb2, n, nvec(i)+1))^(-1/2), ...
        abs(subs(lowera2, n, nvec(i)+1))^(-1/2));
    % lowerbound3(i) = min(abs(subs(lowerb3, n, nvec(i)+1))^(-1/3), ...
    %     abs(subs(lowera3, n, nvec(i)+1))^(-1/3));

    eigcond(i) = cond(VK);
end

figure
set(gcf, 'Position', [200 200 600 350])
loglog(nvec, maxeig .\ upperbound2, '-ok', 'LineWidth', 1, 'MarkerSize', 6, 'MarkerFaceColor', 'w')
hold on
loglog(nvec, mineig .\ lowerbound2 , '-sk', 'LineWidth', 1, 'MarkerSize', 6, 'MarkerFaceColor', 'w')
% loglog(nvec, upperbound2, '-k', 'LineWidth', 1)
% loglog(nvec, lowerbound2, '--k', 'LineWidth', 1)
xlim([nvec(1)/1.2 nvec(end)*1.2])
ylim([0.95 1.15])
xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 14)
ylabel('$|\lambda^b| / |\lambda|$', 'Interpreter', 'latex', 'FontSize', 14)
set(gca, 'FontName', 'Times')
% legend('maximum', 'minimum', 'upper bound', 'lower bound', 'Location', 'northwest', 'FontSize', 12)
legend('minimum', 'maxmum', 'Location', 'southeast', 'FontSize', 12)
exportgraphics(gcf, 'spectral_estimation.pdf')

%% relative true error and increment error
A = readmatrix("TwoErrors.txt");

figure
set(gcf, 'Position', [200 200 600 350])

semilogy(A(:, 2), '-ok', 'LineWidth', 0.8, 'MarkerSize', 6)
hold on
semilogy(A(:, 1), '-*k', 'LineWidth', 0.8, 'MarkerSize', 6)
legend('$\|X_j - X\|_F / \| X \|_F$', '$\|X_j - X_{j-1}\|_F / \| X_j \|_F$', 'Interpreter', 'latex', ...
    'Location', 'northeast', 'FontSize', 12)
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 14)
xlim([0 size(A, 1)+1])
ylabel('Relative error', 'Interpreter', 'latex', 'FontSize', 12)
ylim([min(min(A))/2, max(max(A))*2])
set(gca, 'FontName', 'Times')
exportgraphics(gcf, 'adi_convergence.pdf')

%%  optimality
A = readmatrix("optimal_time.txt");
n = A(:, 1);

figure
set(gcf, 'Position', [200 200 600 350])

loglog(n, A(:, 2), '-sk', 'LineWidth', 1, 'MarkerSize', 8)
hold on
loglog(n, A(:, 3), '-dk', 'LineWidth', 1, 'MarkerSize', 8)
loglog(n, A(:, 4), '-ok', 'LineWidth', 1, 'MarkerSize', 8)
loglog(n, A(:, 5), '-*k', 'LineWidth', 1, 'MarkerSize', 8)

legend('new, $\epsilon=10^{-3}$', 'new, $\epsilon=10^{-6}$', 'new, $\epsilon=10^{-13}$', ...
    'FT, $\epsilon = 10^{-13}$', 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12)
xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('Execution time (s)', 'Interpreter', 'latex', 'FontSize', 12)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)

loglog(n(end-2:end), (n(end-2:end) / n(end)) .^2 * (A(end, 2) / 2), '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off')
loglog(n(end-2:end), (n(end-2:end) .^2 .* log(n(end-2:end)) / (n(end)^2 * log(n(end)))) * (A(end, 5) * 1.5), '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off')
text(4000, 3, '$\mathcal{O}(n^2)$', 'Interpreter', 'latex', 'FontSize', 12)
text(1200, 300, '$\mathcal{O}(n^2 \log n)$', 'Interpreter', 'latex', 'FontSize', 12)

xlim([n(1)/1.1, n(end)*1.1])
ylim([min(min(A(:, 2:5)))/2, max(max(A(:, 2:5)))*2])
yticks(10 .^ (-3:2))
exportgraphics(gcf, 'ex1_time.pdf')
close
%% accuracy
A = readmatrix("optimal_accuracy.txt");
n = A(:, 1);
figure
set(gcf, 'Position', [200 200 600 350])

loglog(n, A(:, 2), '-sk', 'LineWidth', 1, 'MarkerSize', 8)
hold on
loglog(n, A(:, 3), '-dk', 'LineWidth', 1, 'MarkerSize', 8)
loglog(n, A(:, 4), '-ok', 'LineWidth', 1, 'MarkerSize', 8)
xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('$\| X_{new} - X_{FT} \|_{\infty}$', 'Interpreter', 'latex', 'FontSize', 12)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)


xlim([n(1)/1.1, n(end)*1.1])
ylim([min(min(A(:, 6:8)))/10, max(max(A(:, 6:8)))*2])

exportgraphics(gcf, 'ex1_accuracy.pdf')

%% adaptivity to BCs
A = readmatrix("BCs_time.txt");
n = A(:, 1);

figure
set(gcf, 'Position', [200 200 600 350])
loglog(n, A(:, 2), '-ok', 'LineWidth', 1, 'MarkerSize', 8)
hold on
loglog(n, A(:, 3), '-sk', 'LineWidth', 1, 'MarkerSize', 8)
loglog(n, A(:, 4), '-dk', 'LineWidth', 1, 'MarkerSize', 8)
loglog(n, A(:, 5), '-*k', 'LineWidth', 1, 'MarkerSize', 8)

legend('new, $\epsilon=10^{-14}$', 'new, $\epsilon=10^{-3}$', ...
    'new, B-S', 'TO, B-S', 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12)
% legend('new, $\epsilon=10^{-14}$', ...
%     'new, B-S', 'TO, B-S', 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12)
xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('Execution time (s)', 'Interpreter', 'latex', 'FontSize', 12)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)

loglog(n(6:end-1), (n(6:end-1) / n(end-1)) .^3 * (A(end-1, 5) * 1.5), '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off')
loglog(n(6:end), (n(6:end) / n(end)) .^2 * (A(end, 3) / 2), '--k', 'LineWidth', 1.2, 'HandleVisibility', 'off')
text(3000, 1, '$\mathcal{O}(n^2)$', 'Interpreter', 'latex', 'FontSize', 12)
text(1600, 500, '$\mathcal{O}(n^3)$', 'Interpreter', 'latex', 'FontSize', 12)

xlim([n(1)/1.1, n(end)*1.1])
ylim([1e-4, max(max(A(:, 2:5)))*2])

exportgraphics(gcf, 'ex2_time.pdf')

%% accuracy
A = readmatrix("BCs_accuracy.txt");
n = A(:, 1);

figure
set(gcf, 'Position', [200 200 600 350])
loglog(n, A(:, 2), '-ok', 'LineWidth', 1, 'MarkerSize', 8)
hold on
loglog(n, A(:, 3), '-sk', 'LineWidth', 1, 'MarkerSize', 8)
loglog(n, A(:, 4), '-dk', 'LineWidth', 1, 'MarkerSize', 8)
loglog(n, A(:, 5), '-*k', 'LineWidth', 1, 'MarkerSize', 8)

xlabel('$n$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('$\| X - X_{true} \|_{F} / \|X_{true} \|_{F}$', 'Interpreter', 'latex', 'FontSize', 12)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)


xlim([n(1)/1.1, n(end)*1.1])
ylim([5e-16, 1e-3])

exportgraphics(gcf, 'ex2_accuracy.pdf')

%% weak singularity
A = readmatrix("weaksingularity.txt");
n = 1:size(A, 1);

figure
set(gcf, 'Position', [200 200 700 350])
semilogy(n, A(:, 1), '-ok', 'LineWidth', 1, 'MarkerSize', 8)
hold on
semilogy(n, A(:, 2), '-sk', 'LineWidth', 1, 'MarkerSize', 8)
semilogy(n, A(:, 3), '-dk', 'LineWidth', 1, 'MarkerSize', 8)
semilogy(n, A(:, 4), '-*k', 'LineWidth', 1, 'MarkerSize', 8)
semilogy(-1:56, 1e-14 * ones(58, 1), '--k', 'LineWidth', 1, 'HandleVisibility', 'off')

legend('zero, increasing', 'zero, decreasing', 'warm, increasing', ...
    'warm, decreasing', 'Interpreter', 'latex', 'Position', [0.14 0.14 0.47 0.12], 'FontSize', 12, ...
    'NumColumns',2)
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('$\| X_{j} - X_{j-1} \|_{F} / \|X_{j} \|_{F}$', 'Interpreter', 'latex', 'FontSize', 12)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)

xlim([0, n(end)+1])
ylim([1e-20, 2])

exportgraphics(gcf, 'ex3_relerr.pdf')

%% solution
figure
set(gcf, 'Position', [200 200 450 350])
X = readmatrix("weaksingularity_sol.txt");
valX = chebfun2.coeffs2vals(X);
x = chebfun2.chebpts2(size(X, 1), size(X, 2));


surf(x, transpose(x), valX, 'EdgeColor','none')
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)
view(2)
pbaspect([1 1 1])
cmp = colormap('hot');
colormap(flipud(cmp(1:200, :)));
colorbar('FontName', 'Times New Roman', 'FontSize', 10)

exportgraphics(gcf, 'ex3_sol.pdf', 'Resolution', 400)

%% fADI

A = readmatrix("fADI.txt");
n = A(:, 1);

figure
set(gcf, 'Position', [200 200 600 350])
loglog(n, A(:, 2), '-k', 'LineWidth', 1)
hold on
loglog(n, A(:, 4), '--k', 'LineWidth', 1)

legend('fADI', 'ADI', 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12)
xlabel('Iterations', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('Execution time (s)', 'Interpreter', 'latex', 'FontSize', 12)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)

exportgraphics(gcf, 'fadi_time.pdf')

figure
set(gcf, 'Position', [200 200 600 350])
semilogy(n, A(:, 5), '-k', 'LineWidth', 1)
hold on
semilogy(n, A(:, 7), '--k', 'LineWidth', 1)

% legend('fADI', 'ADI', 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 12)
xlabel('$j$', 'Interpreter', 'latex', 'FontSize', 12)
ylabel('$\| X_j - X \|_{F} / \|X \|_{F}$', 'Interpreter', 'latex', 'FontSize', 12)
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12)

exportgraphics(gcf, 'fadi_accuracy.pdf')