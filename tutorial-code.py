
function solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades ){
                      // solve balance of momentum between blade element load and loading in the streamtube
                      // input variables:
                      //     Uinf - wind speed at infinity
                      //     r1_R,r2_R - edges of blade element, in fraction of Radius ;
                      //     rootradius_R, tipradius_R - location of blade root and tip, in fraction of Radius ;
                      //     Radius is the rotor radius
                      //     Omega -rotational velocity
                      //     NBlades - number of blades in rotor

                      // initialize properties of the blade element, variables for output and induction factors
                      var r_R = (r1_R+r2_R)/2; //centroide
                      var Area = Math.PI*(Math.pow(r2_R*Radius,2)-Math.pow(r1_R*Radius,2)); //  area streamtube
                      var a = 0.3; // axial induction factor
                      var anew; // temp new axial induction factor
                      var aline = 0.; // tangential induction factor
                      var Urotor; // axial velocity at rotor
                      var Utan; // tangential velocity at rotor
                      var loads; // normal and tangential loads 2D
                      var load3D = [0 , 0]; // normal and tangential loads 3D
                      var CT; //thrust coefficient at streamtube
                      var Prandtl; // Prandtl tip correction

                      // iteration cycle
                      var Niterations =100; // maximum number of iterations
                      var Erroriterations =0.00001; // error limit for iteration rpocess, in absolute value of induction
                      for (var i = 0; i < Niterations; i++) {

                        ///////////////////////////////////////////////////////////////////////
                        // this is the block "Calculate velocity and loads at blade element"
                        ///////////////////////////////////////////////////////////////////////
                        Urotor = Uinf*(1-a); // axial velocity at rotor
                        Utan = (1+aline)*Omega*r_R*Radius; // tangential velocity at rotor
                        // calculate loads in blade segment in 2D (N/m)
                        loads = loadBladeElement(Urotor, Utan, r_R);
                        load3D[0] =loads[0]*Radius*(r2_R-r1_R)*NBlades; // 3D force in axial direction
                        load3D[1] =loads[1]*Radius*(r2_R-r1_R)*NBlades; // 3D force in azimuthal/tangential direction (not used here)
                        ///////////////////////////////////////////////////////////////////////
                        //the block "Calculate velocity and loads at blade element" is done
                        ///////////////////////////////////////////////////////////////////////

                        ///////////////////////////////////////////////////////////////////////
                        // this is the block "Calculate new estimate of axial and azimuthal induction"
                        ///////////////////////////////////////////////////////////////////////
                        // calculate thrust coefficient at the streamtube
                        CT = load3D[0]/(0.5*Area*Math.pow(Uinf,2));
                        // calculate new axial induction, accounting for Glauert's correction
                        anew = induction_from_thrust_coefficient_Gluert_correction([CT]);
                        // correct new axial induction with Prandtl's correction
                        Prandtl=calculatePrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew);
                        if (Prandtl.Ftotal < 0.0001) { Prandtl.Ftotal = 0.0001; } // avoid divide by zero
                        anew = anew/Prandtl.Ftotal; // correct estimate of axial induction
                        a = 0.75*a+0.25*anew; // for improving convergence, weigh current and previous iteration of axial induction
                        // calculate aximuthal induction
                        aline = loads[1]*NBlades/(2*Math.PI*Uinf*(1-a)*Omega*2*Math.pow(r_R*Radius,2));
                        aline =aline/Prandtl.Ftotal; // correct estimate of azimuthal induction with Prandtl's correction
                        ///////////////////////////////////////////////////////////////////////////
                        // end of the block "Calculate new estimate of axial and azimuthal induction"
                        ///////////////////////////////////////////////////////////////////////

                        // test convergence of solution, by checking convergence of axial induction
                        if (Math.abs(a-anew) < Erroriterations) {
                          i=Niterations; // converged solution, this is the last iteration
                        }

                      };

                      // we have reached a solution or the maximum number of iterations
                      // returns axial induction factor a, azimuthal induction factor a',
                      // and radial position of evaluations and loads
                      return [a , aline, r_R, loads[0] , loads[1]];
                    };