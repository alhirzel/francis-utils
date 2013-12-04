# Alex made this to ensure his implementation is correct. See
# testbench.c:test_bulge_inflate for the implementation.

# after hessenberg decomp but before bulge chasing
A=[2.81014,-6.48911,-15.18678,21.77452,5.07491,-33.61697,7.69500,-3.21216,41.49093,16.75451,37.52472,-4.91861,30.00529;
-79.66993,32.07671,-37.41973,27.72298,3.11131,-59.35638,-17.86218,18.53775,72.09095,3.57814,158.46592,284.03035,333.00127;
0,17.26496,33.19817,-38.30290,-25.19686,31.03556,-58.71160,35.69149,-70.85779,-69.20413,73.02552,-81.81205,-153.14354;
0,0,-4.70294,19.82071,17.59862,-9.54619,22.05326,-23.91293,23.02401,50.87808,-87.43072,41.81270,130.30457;
0,0,0,6.46713,17.41596,-7.30266,-1.34361,0.41457,3.50943,-5.91071,34.33494,15.76940,8.73907;
0,0,0,0,15.48683,20.92098,26.03202,-17.96809,7.73529,46.79353,-112.27704,-9.34287,71.89896;
0,0,0,0,0,11.00435,26.79193,4.71123,-4.04375,-14.29687,16.39901,18.17265,-39.43072;
0,0,0,0,0,0,5.19289,19.11160,3.20148,16.21861,-28.96126,-0.84516,29.94569;
0,0,0,0,0,0,0,7.72795,10.10828,9.25127,-2.09658,-7.79144,3.98539;
0,0,0,0,0,0,0,0,3.19594,12.23062,-9.39841,4.20665,6.76808;
0,0,0,0,0,0,0,0,0,-6.72820,22.71813,-19.60648,-28.18783;
0,0,0,0,0,0,0,0,0,0,-1.11828,25.13190,9.08088;
0,0,0,0,0,0,0,0,0,0,0,-20.40095,-4.33512];

# after bulge chasing but before Schur decomp
M=[	+37.713742,	+2.122909,	-26.879092,	+19.323719,	+24.719264,	+65.097231,	+44.649481,	-18.478583,	-39.066592,	+38.811374,	+232.813101,	-37.580642,	-238.929520;
	-16.743962,	+30.663352,	+32.683726,	+8.812225,	+4.655110,	+99.109491,	-41.663721,	+60.741032,	-102.676764,	+55.579147,	+119.971981,	-190.981198,	-300.525511;
	-0.000000,	+2.473904,	+20.149277,	-0.270192,	-11.148155,	+6.205662,	-7.846479,	+14.239013,	-6.861599,	-4.141649,	-26.229997,	-73.234053,	-29.139398;
	+0.000000,	+0.000000,	+8.978533,	+24.696253,	-10.165111,	-6.989353,	-26.305827,	+13.343400,	-1.641106,	-12.776443,	-54.198215,	-92.748236,	-78.854432;
	+0.000000,	+0.000000,	-0.000000,	+9.372477,	+4.798895,	-12.105243,	-21.835286,	+10.580276,	+3.126046,	-17.352496,	-87.324327,	-102.563512,	-55.919992;
	+0.000000,	+0.000000,	-0.000000,	-4.145944,	+1.895318,	+8.221151,	+5.599537,	-3.744079,	+0.822627,	-0.948806,	+35.700398,	+33.055015,	-35.484344;
	+0.000000,	+0.000000,	+0.000000,	-6.618483,	-3.029363,	+8.253142,	+26.791930,	+4.040804,	-4.093732,	-1.722727,	-3.097368,	-48.487187,	-0.496938;
	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+3.758724,	+13.032328,	-2.161031,	-4.397984,	+9.757796,	+24.969346,	+8.206555;
	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	-0.795916,	+0.227723,	+5.796252,	-7.219318,	-2.535516,	-0.301274,	-13.205336;
	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	-3.493511,	-3.890083,	+0.400003,	+19.002536,	-15.777833,	-29.712108,	-10.384964;
	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	-1.286766,	+10.956775,	+23.224979,	+37.032451;
	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	-0.000000,	+0.000000,	-0.000000,	+8.059128,	+14.060659,	-15.601842;
	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	+0.000000,	-0.000000,	+3.906465,	+22.116861];


N = 13;

disp "Eigenvalues of each matrix should be equal (a bunch of prime numbers):"
eig(A)'
eig(M)'

bbl = 3;         # big bulge location
bbs = N - 3 - 3; # big bulge size

# convert C indices to Octave indices
bbl += 1;
bbs -= 1; # SUSPECT - I don't know why this is needed, make sure C is getting
          # the right matrix slice

subM = M(bbl:(bbl+bbs),bbl:(bbl+bbs))

[Q, S] = schur(subM);

Q
S
