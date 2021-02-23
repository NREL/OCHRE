# -*- coding: utf-8 -*-

import numpy as np


# FUTURE: Convert to Equipment, create stochastic inputs based on self.name (CW, DW, CD)
class WetAppliance(object):
    def __init__(self, Wet_Appliances_Data, App_Name,
                 start_time):  # , B_on, T_uppr, T_lowr, L_used, L_uppr, L_lowr, Bin_uppr, Bin_lowr, P_uppr, P_lowr):
        r"""
        MC Profile Generator
        Generate load profile for a load that has:
            Switch_On Probability Vector
            Specific Demand Profile (e.g. wet appliance)

        """
        self.Set_Profile = Wet_Appliances_Data[App_Name]['PQ_Demand_Profile__2_cols_W_VAr']
        #        self.Switch_On_Prob_Vec=np.kron(np.ones((1,days)),Wet_Appliances_Data[App_Name]['Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440'])
        self.Switch_On_Prob_Vec = Wet_Appliances_Data[App_Name][
            'Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440']
        self.Switch_On_Time_Loc = start_time
        self.Binary_Mem = 0
        self.Binary = 0
        self.Profile_t = 0
        self.P_kW = 0
        self.Q_kVAr = 0
        self.Schedule = 0
        #        self.Bin_test=0
        self.Random_num = 0
        self.Schedule_Finish_Count = 0
        self.Average_Schedule_Delay = Wet_Appliances_Data[App_Name]['Averaged_Scheduled_Delay__Minutes']
        self.Over_ride_start = 0
        self.Over_ride_bin = 0
        self.Schedulable = 0

        # From OCHRE archive, data backend, for calculating event times:
        #  - also see OCHRE archive, bi-modal normal file. Maybe 0-2 events per day?
        #     Dishwasher_Average_Morning_Start = (9)
        #     WashingMachine_Average_Morning_Start = (7.5)
        #     ClothesDryer_Average_Morning_Start = (8)
        #
        #     Dishwasher_Morning_std_dev = 30
        #     WashingMachine_Morning_std_dev = 60
        #     ClothesDryer_Morning_std_dev = 60
        #
        #     Dishwasher_Average_Evening_Start = (19.5)
        #     WashingMachine_Average_Evening_Start = (20)
        #     ClothesDryer_Average_Evening_Start = (21.5)
        #
        #     Dishwasher_Evening_std_dev = 90
        #     WashingMachine_Evening_std_dev = 120
        #     ClothesDryer_Evening_std_dev = 120
        #
        #     # Base on energy per year / cycle energy
        #     # Or consumer behavior data
        #
        #     Dishwasher_Average_Runs_Per_Day = 0.9
        #     WashingMachine_Average_Runs_Per_Day = 0.6
        #     ClothesDryer_Average_Runs_Per_Day = 0.6
        #
        #     # Morning to evening weighting, e.g. 60% run in morning = 0.6
        #
        #     Dishwasher_Morning_Evening_Weight = 0.55
        #     WashingMachine_Morning_Evening_Weight = 0.3
        #     ClothesDryer_Morning_Evening_Weight = 0.6

    def MC_Profile_update(self):
        # From Dwelling code:
        # if self.Appliances_Owned['Washing_Machine'] == 1 and self.Appliances_DR_enabled['Washing_Machine'] == 1:
        #     # TRIGGER ON WAIT TIME OF DW, WM, CD
        #     self.Washing_Machine.Over_ride_start = from_ext_control[
        #         'Washing_Machine_Binary_Start_Override__Binary']
        # if self.Appliances_Owned['Clothes_Dryer'] == 1 and self.Appliances_DR_enabled['Clothes_Dryer'] == 1:
        #     # TRIGGER ON WAIT TIME OF DW, WM, CD
        #     self.Clothes_Dryer.Over_ride_start = from_ext_control[
        #         'Clothes_Dryer_Binary_Start_Override__Binary']
        # if self.Appliances_Owned['Dish_Washer'] == 1 and self.Appliances_DR_enabled['Dish_Washer'] == 1:
        #     # TRIGGER ON WAIT TIME OF DW, WM, CD
        #     self.Dish_Washer.Over_ride_start = from_ext_control[
        #         'Dish_Washer_Binary_Start_Override__Binary']
        #
        # if self.Appliances_Owned['Washing_Machine'] == 1:
        #     IP_ONLINE_CONTROLLER_OP_DEMAND_MODEL[
        #         'Washing_Machine_Remaining_Schedule_time__Minutes'] = self.Washing_Machine.Schedule_Finish_Count
        #     IP_ONLINE_CONTROLLER_OP_DEMAND_MODEL['Washing_Machine_BinaryStatus__Dimensionless'] = (
        #             self.Washing_Machine.Profile_t > 0)
        # if self.Appliances_Owned['Washing_Machine'] == 1:
        #     IP_FORESEE_OP_DEMAND_MODEL[
        #         'Washing_Machine_Remaining_Schedule_time__Minutes'] = self.Washing_Machine.Schedule_Finish_Count
        #     IP_FORESEE_OP_DEMAND_MODEL['Washing_Machine_Remaining_Cycle_time__Minutes'] = (
        #   self.Washing_Machine.Set_Profile.shape[
        #       0] - self.Washing_Machine.Profile_t) * (
        #   self.Washing_Machine.Profile_t > 0)  # full cycle duration if not started; actual remaining time if started; zero if completed
        #     IP_FORESEE_OP_DEMAND_MODEL['Washing_Machine_BinaryStatus__Dimensionless'] = (
        #             self.Washing_Machine.Profile_t > 0)
        #     IP_FORESEE_OP_DEMAND_MODEL['Washing_Machine_Run_Time__Minutes'] = self.Washing_Machine.Set_Profile.shape[0]

        self.Switch_On_Time_Loc = self.MC_SIM_DAY_LOOP(self.Switch_On_Time_Loc)
        if self.Schedulable == 0:
            self.Random_num = np.random.random()
            self.Binary = self.Binary_Mem * 1 + (1 - self.Binary_Mem * 1) * (
                    self.Random_num < self.Switch_On_Prob_Vec[self.Switch_On_Time_Loc,])
            #        self.Bin_test=(self.Random_num<self.Switch_On_Prob_Vec[0,self.Switch_On_Time_Loc])*1
            self.Binary_Mem = self.Binary
            # print self.Binary
            if ((self.Binary > 0) & (self.Profile_t < (self.Set_Profile.shape[0] - 1))):
                self.P_kW = self.Set_Profile[self.Profile_t, 0]
                self.Q_kVAr = self.Set_Profile[self.Profile_t, 1]
                self.Profile_t += 1
            else:
                self.Binary = 0
                self.Profile_t = 0
                self.Binary_Mem = 0
                self.P_kW = 0
                self.Q_kVAr = 0
            self.Switch_On_Time_Loc += 1
        elif self.Schedulable == 1:
            self.Random_num = np.random.random()
            self.Binary = self.Binary_Mem * 1 + (1 - self.Binary_Mem * 1) * (
                    self.Random_num < self.Switch_On_Prob_Vec[self.Switch_On_Time_Loc,])
            if (self.Binary > 0) & (self.Binary_Mem < 1):
                self.Schedule_Finish_Count = -int(
                    (self.Average_Schedule_Delay) * np.log(np.random.random()) - self.Set_Profile.shape[0])
                # print (self.Schedule_Finish_Count)
            if (self.Over_ride_start == 1) or (self.Over_ride_bin == 1):
                self.Over_ride_bin = 1
                self.Schedule_Finish_Count = 0
            #        self.Bin_test=(self.Random_num<self.Switch_On_Prob_Vec[0,self.Switch_On_Time_Loc])*1
            self.Binary_Mem = self.Binary
            # print self.Binary
            if ((self.Binary > 0) & (self.Profile_t < (self.Set_Profile.shape[0] - 1)) & (
                    self.Schedule_Finish_Count == 0 or self.Over_ride_bin == 1)):
                self.P_kW = self.Set_Profile[self.Profile_t, 0]
                self.Q_kVAr = self.Set_Profile[self.Profile_t, 1]
                self.Profile_t += 1
            elif ((self.Binary > 0) & (self.Profile_t < (self.Set_Profile.shape[0] - 1)) & (
                    self.Schedule_Finish_Count > 0 or self.Over_ride_bin == 0)):
                self.Profile_t = 0
                self.P_kW = 0
                self.Q_kVAr = 0
                self.Schedule_Finish_Count -= 1
            else:
                self.Binary = 0
                self.Profile_t = 0
                self.Binary_Mem = 0
                self.P_kW = 0
                self.Q_kVAr = 0
                self.Over_ride_bin = 0
                self.Over_ride_start = 0
            self.Switch_On_Time_Loc += 1

    def MC_SIM_DAY_LOOP(self, vec_day_minute):
        if (vec_day_minute == 1440):
            vec_day_minute = 0
        return (vec_day_minute)
