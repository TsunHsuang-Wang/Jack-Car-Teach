from scipy.stats import poisson

RENT_EXPECT_A = 3
RENT_EXPECT_B = 4
RETURN_EXPECT_A = 3
RETURN_EXPECT_B = 2

pois_A_rent = poisson(RENT_EXPECT_A)
pois_A_return = poisson(RETURN_EXPECT_A)
pois_B_rent = poisson(RENT_EXPECT_B)
pois_B_return = poisson(RETURN_EXPECT_B)

# distribution of car rental in location A
def A_rent_prob(n):
	p = pois_A_rent.pmf(n)
	return p
# distribution of car return in location A
def A_return_prob(n):
	p = pois_A_return.pmf(n)
	return p
# distribution of car rent in location B
def B_rent_prob(n):
	p = pois_B_rent.pmf(n)
	return p
# distribution of car return in location B
def B_return_prob(n):
	p = pois_B_return.pmf(n)
	return p

