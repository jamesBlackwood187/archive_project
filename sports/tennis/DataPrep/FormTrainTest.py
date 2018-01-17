import pandas as pd 
import numpy as np 


def GetStats(row, wonMatchSet, loseMatchSet):
	top20Wins    =  wonMatchSet[wonMatchSet['loser_rank'] <= 20]
	top20Losses  =  loseMatchSet[loseMatchSet['winner_rank'] <= 20]
	surfaceWins  =  wonMatchSet[wonMatchSet['surface'] == row['surface']]
	surfaceLosses  =  loseMatchSet[loseMatchSet['surface'] == row['surface']]

	carreerHistory = wonMatchSet.append(loseMatchSet, ignore_index = True)

	if (len(wonMatchSet.index) + len(loseMatchSet.index) != 0):
	    playerWinPct = float(len(wonMatchSet.index)) / (len(wonMatchSet.index) + len(loseMatchSet.index))
	    avgSets = float((wonMatchSet['NumSets'].sum() + loseMatchSet['NumSets'].sum()) )/ (len(wonMatchSet.index) + len(loseMatchSet.index))
	    avgSetsPerWin = np.mean(wonMatchSet['NumSets'])
	    avgSetsPerLoss = np.mean(loseMatchSet['NumSets'])

	else:
		playerWinPct = np.nan
		avgSets = np.nan
		avgSetsPerWin = np.nan
		avgSetsPerLoss = np.nan
		
	    
	if (len(top20Losses.index) + len(top20Wins.index) != 0):
	    top20WinPct  =  float(len(top20Wins.index)) / (len(top20Losses.index) + len(top20Wins.index))
	else:
		top20WinPct = np.nan


	if (len(surfaceWins.index) + len(surfaceLosses.index) != 0):
	    surfaceWinPctVsCarrerWinPct  =  (float(len(surfaceWins.index)) / (len(surfaceWins.index) + len(surfaceLosses.index)))  - playerWinPct 
	else:
	    surfaceWinPctVsCarrerWinPct = np.nan


	if (wonMatchSet['NumSets'].sum() > 0 and loseMatchSet['NumSets'].sum() > 0):
		acesInWins  = wonMatchSet['w_ace'].sum()
		acesInLosses = loseMatchSet['l_ace'].sum()
		acesPerSet = float((acesInWins + acesInLosses)) / carreerHistory['NumSets'].sum()

		dfInWins  = wonMatchSet['w_df'].sum()
		dfInLosses = loseMatchSet['l_df'].sum()
		dfPerSet = float((dfInWins + dfInLosses)) / carreerHistory['NumSets'].sum()

		svptInWins  = wonMatchSet['w_svpt'].sum()
		svptInLosses = loseMatchSet['l_svpt'].sum()
		svptPerSet = float((svptInLosses + svptInWins)) / carreerHistory['NumSets'].sum()

		fInInWins  = wonMatchSet['w_1stIn'].sum()
		fInInLosses = loseMatchSet['l_1stIn'].sum()
		fInPerSet  = float((fInInLosses+fInInWins)) / carreerHistory['NumSets'].sum()

		fWonInWins  = wonMatchSet['w_1stWon'].sum()
		fWonInLosses = loseMatchSet['l_1stWon'].sum()
		fWonPerSet  = float((fWonInLosses+fWonInWins)) / carreerHistory['NumSets'].sum()

		sWonInWins  = wonMatchSet['w_2ndWon'].sum()
		sWonInLosses = loseMatchSet['l_2ndWon'].sum()
		sWonPerSet  = float((sWonInLosses+sWonInWins)) / carreerHistory['NumSets'].sum()


		svGameInWins  = wonMatchSet['w_SvGms'].sum()
		svGameInLosses = loseMatchSet['l_SvGms'].sum()
		svGamePerSet  = float((svGameInLosses+svGameInWins)) / carreerHistory['NumSets'].sum()

		bpsInWins  = wonMatchSet['w_bpSaved'].sum()
		bpsInLosses = loseMatchSet['l_bpSaved'].sum()
		bpsPerSet  = float((bpsInLosses+bpsInWins)) / carreerHistory['NumSets'].sum()

		bpfInWins  = wonMatchSet['w_bpFaced'].sum()
		bpfInLosses = loseMatchSet['l_bpFaced'].sum()
		bpfPerSet  = float((bpfInLosses+bpfInWins)) / carreerHistory['NumSets'].sum()


	else:
		acesPerSet = np.nan
		dfPerSet = np.nan
		svptPerSet = np.nan
		fInPerSet = np.nan
		fWonPerSet = np.nan
		sWonPerSet = np.nan
		svGamePerSet = np.nan
		bpsPerSet = np.nan
		bpfPerSet = np.nan
	return (playerWinPct, top20WinPct, avgSets, avgSetsPerWin, avgSetsPerLoss, surfaceWinPctVsCarrerWinPct, acesPerSet, dfPerSet, svptPerSet, fInPerSet, fWonPerSet, sWonPerSet, svGamePerSet, bpsPerSet, bpfPerSet)


def GetPlayerFeatures(df, row, player_id, currentMatch):
	playerWonMatches = df[(df['winner_id'] == player_id) & (df.index < currentMatch)]
	playerLoseMatches = df[(df['loser_id'] == player_id) & (df.index < currentMatch)]
	playerStats = GetStats(row, playerWonMatches, playerLoseMatches)
	return playerStats

def FormDataSet(df):
	dfC = pd.DataFrame()
	for j,row in df.iterrows():
		if(row['loser_rank'] > row['winner_rank']):
			player1ID = row['loser_id']
			player2ID = row['winner_id']


			player1Ht = row['loser_ht']
			player2Ht = row['winner_ht']

			player1Hand = row['loser_hand']
			player2Hand = row['winner_hand']

			player1Age = row['loser_age']
			player2Age = row['winner_age']

			player1Rank = row['loser_rank']
			player2Rank = row['winner_rank']

			target = 0
		else:
			player1ID = row['winner_id']
			player2ID = row['loser_id']
			
			player1Ht = row['winner_ht']
			player2Ht = row['loser_ht']

			player1Hand = row['winner_hand']
			player2Hand = row['loser_hand']

			player1Age = row['winner_age']
			player2Age = row['loser_age']

			player1Rank = row['winner_rank']
			player2Rank = row['loser_rank']

			target = 1
		if row['tourney_name'] in ['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open']:
			major = 1
		else:
			major = 0

		playerWinPct_p1, top20WinPct_p1, avgSets_p1, avgSetsPerWin_p1, avgSetsPerLoss_p1, surfaceWinPctVsCarrerWinPct_p1, acesPerSet_p1, dfPerSet_p1, svptPerSet_p1, fInPerSet_p1, fWonPerSet_p1, sWonPerSet_p1, svGamePerSet_p1, bpsPerSet_p1, bpfPerSet_p1 = GetPlayerFeatures(df, row, player1ID,j)
		playerWinPct_p2, top20WinPct_p2, avgSets_p2, avgSetsPerWin_p2, avgSetsPerLoss_p2, surfaceWinPctVsCarrerWinPct_p2, acesPerSet_p2, dfPerSet_p2, svptPerSet_p2, fInPerSet_p2, fWonPerSet_p2, sWonPerSet_p2, svGamePerSet_p2, bpsPerSet_p2, bpfPerSet_p2 = GetPlayerFeatures(df, row, player2ID,j)


		data_row = {'game':j, 'Target':target
					, 'surface': row['surface']
					, 'date'   : row['tourney_date']
					, 'major'  : major 
					,'player1Rank':player1Rank, 'player2Rank': player2Rank
					,'player1Height':player1Ht, 'player2Height': player2Ht
					,'player1Age':player1Age, 'player2Age': player2Age
					,'player1Hand':player1Hand, 'player2Hand': player2Hand
					, 'playerWinPct_p1':playerWinPct_p1
					, 'top20WinPct_p1' :top20WinPct_p1
					, 'avgSets_p1': avgSets_p1
					, 'avgSetsPerWin_p1':avgSetsPerWin_p1
					, 'avgSetsPerLoss_p1':avgSetsPerLoss_p1
					, 'surfaceWinPctVsCarrerWinPct_p1':surfaceWinPctVsCarrerWinPct_p1
					, 'acesPerSet_p1': acesPerSet_p1  
					, 'dfPerSet_p1': dfPerSet_p1
					, 'svptPerSet_p1':svptPerSet_p1
					, 'fInPerSet_p1':fInPerSet_p1
					, 'fWonPerSet_p1':fWonPerSet_p1
					, 'sWonPerSet_p1':sWonPerSet_p1
					, 'svGamePerSet_p1':svGamePerSet_p1
					, 'bpsPerSet_p1':bpsPerSet_p1
					,  'bpfPerSet_p1':bpfPerSet_p1 
					, 'playerWinPct_p2':playerWinPct_p2
					, 'top20WinPct_p2' :top20WinPct_p2
					, 'avgSets_p2': avgSets_p2
					, 'avgSetsPerWin_p2':avgSetsPerWin_p2
					, 'avgSetsPerLoss_p2':avgSetsPerLoss_p2
					, 'surfaceWinPctVsCarrerWinPct_p2':surfaceWinPctVsCarrerWinPct_p2
					, 'acesPerSet_p2': acesPerSet_p2  
					, 'dfPerSet_p2': dfPerSet_p2
					, 'svptPerSet_p2':svptPerSet_p2
					, 'fInPerSet_p2':fInPerSet_p2
					, 'fWonPerSet_p2':fWonPerSet_p2
					, 'sWonPerSet_p2':sWonPerSet_p2
					, 'svGamePerSet_p1':svGamePerSet_p2
					, 'bpsPerSet_p2':bpsPerSet_p2
					,  'bpfPerSet_p2':bpfPerSet_p2
					}

		data_rowDF = pd.DataFrame(data_row, index = ['game'])
		dfC = dfC.append(data_row, ignore_index = True)
	return dfC


if __name__ == '__main__':
	file = '/home/jbl/Documents/Programming/Sports Betting/tennis/Data/fullMatches.csv'
	file2 = '/home/jbl/Documents/Programming/Sports Betting/tennis/Data/frenchOpen1.csv'
	fullDFs = pd.read_csv(file, index_col = 'ix')
	testDf = pd.read_csv(file2, index_col = 'ix')

	fullDFs =fullDFs.append(testDf)

	fullDFs['StringScore'] = fullDFs['score'].astype(str)
	fullDFs['NumSets']     = fullDFs['StringScore'].apply(lambda x: x.count('-'))


	FeatureSet = FormDataSet(fullDFs)

	df_p1_hand = pd.get_dummies(FeatureSet['player1Hand'])
	df_p2_hand = pd.get_dummies(FeatureSet['player2Hand'])
	df_surface = pd.get_dummies(FeatureSet['surface'])

	FeatureSet = pd.concat([FeatureSet, df_p1_hand, df_p2_hand, df_surface], axis = 1)
	FeatureSet['NanCount'] = FeatureSet.isnull().sum(axis=1)

	FeatureSet = FeatureSet.fillna(-1)
	FeatureSet.drop(['date', 'game', 'player1Hand', 'player2Hand'])
	a = FeatureSet.to_csv('/home/jbl/Documents/Programming/Sports Betting/tennis/Data/FeatureSet.csv')

