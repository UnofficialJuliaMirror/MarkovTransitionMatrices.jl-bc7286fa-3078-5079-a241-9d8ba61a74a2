


if Pcu==inf; ZPu = 100000; else; ZPu = (log(Pcu) - log(p) - muP + v^2/2) / v; end
if Pcl==-inf; ZPl = -100000; else; ZPl = (log(Pcl) - log(p) - muP + v^2/2) / v; end

if DRcu==inf; ZDu = 100000; else; ZDu = (log(DRcu) - log(d) - muD + vD^2/2) / vD; end
if DRcl==-inf; ZDl = -100000; else; ZDl = (log(DRcl) - log(d) - muD + vD^2/2) / vD; end
Prob = bvnl(ZPu,ZDu,rho) - bvnl(ZPl,ZDu,rho) - bvnl(ZPu,ZDl,rho) + bvnl(ZPl,ZDl,rho);


function bvnl(xlim, ylim, ρ)
    xl,xu = xlim
    yl,yu = ylim
    xl < xu && yl < yu || throw(DomainError())
    return bvnuppercdf(xu,yu,ρ) - bvnuppercdf(xl,yu,ρ) - bvnuppercdf(xu,yl,ρ) + bvnuppercdf(xl,yl,ρ)
end
