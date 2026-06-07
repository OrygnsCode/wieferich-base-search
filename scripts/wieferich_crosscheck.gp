\\ Independent Wieferich cross-check in PARI/GP (one-liner loops).
\\ PARI/GP is a separate math engine from our Python/CUDA stack and is the
\\ language OEIS editors use to verify submissions. Wieferich condition:
\\ b^(p-1) == 1 (mod p^2).
default(parisize, 500000000);
is_wief(b,p) = (Mod(b,p^2)^(p-1) == 1);

print("=== PART 1: reproduce known smallest Wieferich primes (A039951) ===");
known = [[2,1093],[3,11],[5,2],[7,5],[11,71],[12,2693],[13,2],[20,281]];
ok1 = 1;
for(k=1, #known, my(b=known[k][1], t=known[k][2], f=0); forprime(q=2, t, if(is_wief(b,q), f=q; break)); if(f!=t, ok1=0); print("  base ", b, ": smallest = ", f, "  (expected ", t, ")  ", if(f==t,"OK","FAIL")));
print("PART 1 verdict: ", if(ok1, "ALL OK", "FAILURE"));
print("");

print("=== PART 2: positive control, known base-941 prime at ~6.45e13 ===");
p941 = 64501672625861;
r941 = is_wief(941, p941);
print("  is_wief(941, ", p941, ") = ", r941, "  (expect 1)");
print("PART 2 verdict: ", if(r941==1, "OK", "FAIL"));
print("");

print("=== PART 3: negative cross-check, bases 186,187,200 over [4.4e12, +1e6] ===");
lo = 4400000000000; hi = 4400001000000;
tb = [186, 187, 200];
total = 0; nprimes = 0;
for(k=1, #tb, my(b=tb[k], cnt=0); forprime(q=lo, hi, if(b==tb[1], nprimes++); if(is_wief(b,q), print("  FOUND base ", b, " p=", q); cnt++)); total += cnt; print("  base ", b, ": ", cnt, " Wieferich primes in window"));
print("PART 3 verdict: ", if(total==0, "0 found (agrees with GPU pipeline)", "MISMATCH"));
print("  primes scanned per base in window: ", nprimes);
quit;
